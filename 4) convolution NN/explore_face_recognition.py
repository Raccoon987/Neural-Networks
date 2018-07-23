from preprocess import getData, getImageData, init_weight_and_bias, init_filter, y_hot_encoding, error_rate
from conv_tf_face_recognition import ConvNeuralNetwork_Image, ConvPullLayer, HiddenLayerBatchNorm, HiddenLayer

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import pandas as pd
from itertools import product
import csv
from os import walk
import os


f = lambda f_name: os.path.realpath(os.path.join(os.getcwd(), f_name)).replace("\\", "/")

def explore_data():
    X, Y = getData()
    print(X.shape)

def main():
    ''' data shape: (35887, 2304)'''
    X, Y = getImageData()
    X, Y = X[:5000], Y[:5000]
    # reshape X for tf: N x w x h x c
    X = X.transpose((0, 2, 3, 1))

    model = ConvNeuralNetwork_Image(convpull_layer_sizes=((20, 5, 5), (50, 5, 5)), conv_nonlin_functions=("tanh", "tanh"), poolsz=[2, 2],
                                    hidden_layer_sizes=[500, 300], nonlin_functions=("relu", "relu"), dropout_coef=(0.9, 0.8, 0.8))
    session = tf.InteractiveSession()
    model.set_session(session)
    model.fit(X, Y, optimizer="adam", optimizer_params=(10e-4, 0.99, 0.999), reg=10e-3, epochs=10, batch_size=1000, split=True, show_fig=True, print_every=10, print_tofile=False)
    #model.fit(X, Y, optimizer="adam", learning_rate=10e-4, mu=0.99, decay=0.999, reg=10e-3, epochs=10, batch_size=1000, split=True, show_fig=True, print_every=10)

    ''' #train_op = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0).minimize(cost)
        #train_op = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999).minimize(cost)
        #train_op = tf.train.MomentumOptimizer(learning_rate, momentum=mu, use_locking=False, name='Momentum', use_nesterov=False).minimize(cost)
        #train_op = tf.train.ProximalGradientDescentOptimizer(learning_rate, l1_regularization_strength=0.0, l2_regularization_strength=0.0, use_locking=False).minimize(cost) '''



def optimizer_learnrate_search():
    ''' choose the best optimizer and best lerning rate '''
    optimizers_lst = ["adam", "rmsprop", "momentum", "proximal"]
    learnrate_lst = [10e-5, 10e-4, 10e-3, 10e-2]

    X, Y = getImageData()
    X = X.transpose((0, 2, 3, 1))

    model = ConvNeuralNetwork_Image(convpull_layer_sizes=((20, 5, 5), (50, 5, 5)), conv_nonlin_functions=("tanh", "tanh"), poolsz=[2, 2],
                                    hidden_layer_sizes=[500, 300], nonlin_functions=("relu", "relu"), dropout_coef=(0.9, 0.8, 0.8))
    session = tf.InteractiveSession()
    model.set_session(session)

    for opt in optimizers_lst:
        for lrn_rate in learnrate_lst:
            if opt == "momentum":
                for mu in [0.9, ]:
                    print("START optimizer: {} learnrate: {} momentun: {}".format(opt, str(lrn_rate), str(mu)))
                    model.fit(X, Y, optimizer=opt, optimizer_params=(lrn_rate, mu, ), reg=10e-3, epochs=10, batch_size=1000, split=True, show_fig=False, print_every=10,
                              print_tofile=f("1step) optimizer/" + opt + str(lrn_rate) + str(mu) + ".csv"))
            else:
                print("START optimizer: {} learnrate: {}".format(opt, str(lrn_rate)))
                model.fit(X, Y, optimizer=opt, optimizer_params=(lrn_rate, ), reg=10e-3, epochs=10, batch_size=1000, split=True, show_fig=False, print_every=10,
                          print_tofile=f("1step) optimizer/" + opt + str(lrn_rate) + ".csv"))



def plot_(path):
    file_lst = []
    ''' path = f("conv_face_grid_search/1step) optimizer/") 
               f("conv_face_grid_search/2step) convlayer2/")
               f("conv_face_grid_search/2step) convlayer3/")
               f("conv_face_grid_search/3step) vanila_layers/")
    '''
    for (dirpath, dirnames, filenames) in walk(path):
        file_lst.extend([fl for fl in filenames if (os.path.splitext(fl)[1] != ".png")])
        break

    print(file_lst)
    ''' plot costs '''

    for num in range(len(file_lst)):
        f = pd.read_csv(path + file_lst[num])
        plt.subplot(2, 4, num+1)
        plt.plot(list(map(float, f.iloc[0].index)), color="blue", label="batch_costs")
        plt.plot(list(f.iloc[0]), color="red", label="valid_costs")
        plt.title(file_lst[num], fontsize=8)
        plt.legend(loc='best')
        plt.grid()
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.2, hspace=0.2)
    plt.show()

    for num in range(len(file_lst)):
        f = pd.read_csv(path + file_lst[num])
        plt.subplot(2, 4, num+1)
        plt.plot(list(f.iloc[1]), color="red", label="error_rate")
        plt.title(file_lst[num], fontsize=8)
        plt.legend(loc='best')
        ax = plt.gca()  # grab the current axis
        ax.set_yticks(np.arange(0.4, 1.0, 0.1))
        plt.grid()
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.2, hspace=0.2)
    plt.show()


def grid_search_one():

    #conv_sz_lst = [((20, 5, 5), (50, 5, 5)),   ((20, 3, 3), (50, 3, 3)),   ((20, 7, 7), (50, 7, 7))]
    #conv_nonlin_lst = ["tanh", "relu"]

    ''' RELU RELU IS THE BEST FROM THE DOUBLE CONVOLUTION LAYERS INSPECTION  '''

    conv_sz_lst = [((20, 5, 5), (50, 5, 5), (80, 5, 5)), ((20, 3, 3), (50, 3, 3), (80, 3, 3)),   ((15, 5, 5), (25, 5, 5), (50, 5, 5)),  ((15, 7, 7), (25, 7, 7), (50, 7, 7)),
                   ((20, 7, 7), (50, 7, 7), (80, 7, 7))]
    conv_nonlin_lst = ["relu"]
    ''' (20, 3, 3)_(50, 3, 3)_(80, 3, 3)relu_relu_relu IS THE BEST - CLOSIEST ERR_RATE TO 0.4'''

    X, Y = getImageData()
    X = X.transpose((0, 2, 3, 1))

    for size in conv_sz_lst:
        for func in product(conv_nonlin_lst, repeat=len(size)):
            print(size, func)
            model = ConvNeuralNetwork_Image(convpull_layer_sizes=size, conv_nonlin_functions=func,
                                            poolsz=[2, 2], hidden_layer_sizes=[500, 300], nonlin_functions=("relu", "relu"), dropout_coef=(0.9, 0.8, 0.8))
            session = tf.InteractiveSession()
            model.set_session(session)
            model.fit(X, Y, optimizer="adam", optimizer_params=(10e-4, ), reg=10e-3, epochs=10, batch_size=1000, split=True, show_fig=False, print_every=10,
                      print_tofile=f("2step) convlayer2/" + "adam" + "_".join(map(str, size)) + "_".join(func)))

def grid_search_two():
    hidd_sz_lst = [[500, 300],  [500, 300, 80],  [1000],  [300]]
    hidd_nonlin_lst = ["tanh", "relu"]
    dropout_lst = [0.8, 0.6]

    X, Y = getImageData()
    X = X.transpose((0, 2, 3, 1))

    for size in hidd_sz_lst:
        for func in product(hidd_nonlin_lst, repeat=len(size)):
            for drop in dropout_lst:
                model = ConvNeuralNetwork_Image(convpull_layer_sizes=((20, 3, 3), (50, 3, 3), (80, 3, 3)), conv_nonlin_functions=("relu", "relu", "relu"), poolsz=[2, 2],
                                                hidden_layer_sizes=size, nonlin_functions=func, dropout_coef=[drop]*(len(size)+1))
                session = tf.InteractiveSession()
                model.set_session(session)
                model.fit(X, Y, optimizer="adam", optimizer_params=(10e-4, ), reg=10e-3, epochs=10, batch_size=1000, split=True, show_fig=False, print_every=10,
                          print_tofile=f("3step) vanila_layers/" + "_".join(map(str, size)) + "_".join(func) + str(drop) + ".csv"))



#main()
#explore_data()
#optimizer_learnrate_search()
#plot_(f("optimizer/"))
#grid_search_one()
#plot_(f("convlayer3/"))
#grid_search_two()
#plot_(f("vanila_layers/"))


''' BEST PARAMETERS: optimizer="adam"                                              optimizer_params=(10e-4, )                          reg=10e-3 
                     convpull_layer_sizes=((20, 3, 3), (50, 3, 3), (80, 3, 3))     conv_nonlin_functions=("relu", "relu", "relu")      poolsz=[2, 2]
                     hidden_layer_sizes=[1000]                                     nonlin_functions=["relu"]                           dropout_coef=[0.6, 0.6]
                     OR
                     hidden_layer_sizes=[500, 300]                                 nonlin_functions=["tanh", "tanh"]                   dropout_coef=[0.8, 0.8, 0.8]
'''


def grid_search_three():
    optimizers_lst = ["adam", "rmsprop", "momentum", "proximal"]
    learnrate_lst = [10e-5, 10e-4, 10e-3]

    X, Y = getImageData()
    X = X.transpose((0, 2, 3, 1))

    model = ConvNeuralNetwork_Image(convpull_layer_sizes=((20, 3, 3), (50, 3, 3), (80, 3, 3)), conv_nonlin_functions=("relu", "relu", "relu"), poolsz=[2, 2],
                                    hidden_layer_sizes=[500, 300], nonlin_functions=("tanh", "tanh"), dropout_coef=(0.8, 0.8, 0.8))
    session = tf.InteractiveSession()
    model.set_session(session)

    for opt in optimizers_lst:
        for lrn_rate in learnrate_lst:
            if opt == "momentum":
                for mu in [0.9, ]:
                    print("START optimizer: {} learnrate: {} momentun: {}".format(opt, str(lrn_rate), str(mu)))
                    model.fit(X, Y, optimizer=opt, optimizer_params=(lrn_rate, mu,), reg=10e-3, epochs=12, batch_size=1000, split=True, show_fig=False, print_every=3,
                              print_tofile=f("1step) optimizer/" + opt + str(lrn_rate) + str(mu) + ".csv"))

            else:
                print("START optimizer: {} learnrate: {}".format(opt, str(lrn_rate)))
                model.fit(X, Y, optimizer=opt, optimizer_params=(lrn_rate,), reg=10e-3, epochs=12, batch_size=1000, split=True, show_fig=False, print_every=3,
                          print_tofile=f("1step) optimizer/" + opt + str(lrn_rate) + ".csv"))



#grid_search_three()
#plot_(f("1step) optimizer/"))
''' THE BEST OPTIMIZER - for:
    hidden_layer_sizes=[500, 300]                                 nonlin_functions=["tanh", "tanh"]                   dropout_coef=[0.8, 0.8, 0.8]
    STILL:
    adam   learnrate = 10e-4                                      and 2 place: momentum lnrate = 10e-3, mu = 0.9                               '''






def grid_search_atricle():
    hidd_nonlin_lst = ["tanh", "relu"]

    X, Y = getImageData()
    X = X.transpose((0, 2, 3, 1))

    for conv_func in (("tanh", "tanh", "tanh"), ("relu", "relu", "relu")):
        for vanila_func in product(hidd_nonlin_lst, repeat=2):

            model = ConvNeuralNetwork_Image(convpull_layer_sizes=((10, 5, 5), (10, 5, 5), (10, 3, 3)), conv_nonlin_functions=conv_func, poolsz=[2, 2],
                                            hidden_layer_sizes=[256, 128], nonlin_functions=vanila_func, dropout_coef=(1.0, 0.5, 0.5))
            session = tf.InteractiveSession()
            model.set_session(session)
            model.fit(X, Y, optimizer="adam", optimizer_params=(10e-3, ), reg=10e-3, epochs=10, batch_size=1000, split=True, show_fig=False, print_every=10,
                      print_tofile=f("nonlinear/" + "Conv " + "_".join(conv_func) + " fully " + "_".join(vanila_func) + ".csv"))

#grid_search_atricle()
#plot_(f("nonlinear/"))

#fl = pd.read_csv("C:/Users/Alex/PycharmProjects/Theano_Tensorflow/Convolution_networks/conv_face_grid_search/convlayers/adam(20, 3, 3)_(50, 3, 3)relu_relu")
#print(fl.shape)
#print(fl.iloc[0])
#print(fl.iloc[1])
#print(list(map(float, fl.iloc[0].index)))