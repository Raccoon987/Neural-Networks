import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
import pandas as pd
import sys
import os

from sklearn.utils import shuffle
from autoencoders import AutoEncoder, RestBolzmanMachine, ConvolveAutoEncoder, DeepNeuralNetwork


f = lambda f_name: os.path.realpath(os.path.join(os.getcwd(), f_name)).replace("\\", "/")


def getData(balance_ones=False):
    # images are 48x48 = 2304 size vectors;  N = 35887
    Y, X = [], []
    first = True
    for line in open(f('/../4) convolution NN/fer2013.csv')):
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

def y_hot_encoding(y):
    N = len(y)
    K = len(set(y))
    matrix = np.zeros((N, K))
    for i in range(N):
        matrix[i, y[i]] = 1
    return matrix

def error_rate(targets, predictions):
    return np.mean(targets != predictions)

# filter shapes are expected to be: filter width x filter height x input feature maps x output feature maps
# ??? sqrt( height*weight*color + feature*heiht*weight/(2*2)) ??? why so?
def init_filter(shape, poolsz):
    w = np.random.randn(*shape) * np.sqrt(2) / np.sqrt(np.prod(shape[:-1]) + shape[-1]*np.prod(shape[:-2] / np.prod(poolsz)))
    return w.astype(np.float32), np.zeros(shape[-1], dtype=np.float32)



def explore_data():
    train = shuffle(pd.read_csv(f("mnist_train.csv")).as_matrix().astype(np.float32))
    X, Y = train[:, 1:] / 255, train[:, 0].astype(np.int32)
    print(X.shape)


def grid_single_auto_search(M2, nonlin_func, auto_optimizer, auto_opt_args, dropout):
    img_lst = np.random.randint(999, size=4)
    for m2 in M2:
        for f in nonlin_func:
            for opt in auto_optimizer:
                for opt_args in auto_opt_args:
                    for drop in dropout:
                        if opt_args[0] >= 1e-2:
                            epoch = 3
                        else:
                            epoch = 5
                        explore_single_autoencoder(m2, f, 1, opt, opt_args, drop, epoch, img_lst)


def explore_single_autoencoder(M2, encoder_type, nonlin_func, id, auto_optimizer, auto_opt_args, dropout, epoch, img_lst):
    ''' M2 = int number =>  AutoEncoder or RestBolzmanMachine
        M2 = (in_fm, out_fm, width, height) => ConvolveAutoEncoder
        encoder_type : {"AutoEncoder", "RestBolzmanMachine", "ConvolveAutoEncoder"} '''

    ''' MNIST 28x28 '''
    #train = shuffle(pd.read_csv(f("mnist_train.csv")).as_matrix().astype(np.float32))
    #X, Y = train[:, 1:] / 255, train[:, 0].astype(np.int32)

    ''' FACE EMOTION RECOGNITION '''
    X, Y = getImageData()

    encoder_dict = {"AutoEncoder" : AutoEncoder, "RestBolzmanMachine" : RestBolzmanMachine, "ConvolveAutoEncoder" : ConvolveAutoEncoder}

    ''' TRAIN SINGLE AUTOENCODER '''
    if encoder_type == "AutoEncoder" or encoder_type == "RestBolzmanMachine":
        ''' shape of X faces is (35887, 1, 48, 48) '''
        X = X.reshape(X.shape[0], np.prod(X.shape[1:]))
        print("START: " + str(M2) + "_" + nonlin_func + "_" + auto_optimizer + "_".join(map(str, auto_opt_args)) + str(dropout))
        ''' M1, M2, nonlin_func, id, auto_optimizer, auto_opt_args, dropout '''
        encoder = encoder_dict[encoder_type](X.shape[1], M2, nonlin_func, id, auto_optimizer, auto_opt_args, dropout)
    elif encoder_type == "ConvolveAutoEncoder":
        ''' reshape X for tf: N x w x h x c '''
        X = X.transpose((0, 2, 3, 1))
        encoder = encoder_dict[encoder_type](M2[0], M2[1], M2[2], M2[3], nonlin_func, id, auto_optimizer, auto_opt_args)

    session = tf.InteractiveSession()
    encoder.set_session(session)
    encoder.fit(X[:-1000], epochs=epoch, show_fig=False)


    """ GET IMAGE PREDICTION """
    fig = plt.figure(figsize=(12, 10))
    outer = gridspec.GridSpec(2, 2, wspace=0.2, hspace=0.2)
    for i in range(len(img_lst)):
        face = X[-1000:][i]
        image_lst = [face, encoder.predict(np.array([face]))]
        inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[i], wspace=0.1, hspace=0.1)
        for j in range(2):
            ax = plt.Subplot(fig, inner[j])
            ax.imshow(image_lst[j].reshape(48, 48), cmap='gray')
            fig.add_subplot(ax)
    #plt.show()
    #fig.show()
    if isinstance(M2, tuple):
        name = '_'.join(list(map(str, M2))) + "_" + nonlin_func + "_" + auto_optimizer + "_".join(map(str, auto_opt_args)) + str(dropout)
    fig.savefig(name + ".png")
    plt.close(fig)

    '''
    done = False
    while not done:
        i = np.random.choice(len(X[-1000:]))
        x = X_[-1000:][i]
        y= autoencoder.predict([x])
        plt.subplot(1, 3, 1)
        plt.imshow(x.reshape(48, 48), cmap='gray')
        plt.title('Original')

        plt.subplot(1, 3, 2)
        plt.imshow(y.reshape(48, 48), cmap='gray')
        plt.title('Reconstructed drop=0.6')

        plt.show()

        ans = input("Generate another?")
        if ans and ans[0] in ('n' or 'N'):
            done = True
    '''

def deepnetwork():
    '''mnist_train has 42000 elements, each element - 784 parameters'''
    #train = shuffle(pd.read_csv("mnist_train.csv").as_matrix().astype(np.float32))
    #X, Y = train[:, 1:] / 255, train[:, 0].astype(np.int32)

    ''' shape of X faces is (35887, 1, 48, 48) '''
    X, Y = getImageData()
    X_ = X.reshape(X.shape[0], np.prod(X.shape[1:]))

    dnn = DeepNeuralNetwork(auto_hid_layer_sz=[2000, 1400, 800, 400, 100], 
			    auto_nonlin_func=["sigmoid", "sigmoid", "sigmoid", "sigmoid", "sigmoid"], 
			    auto_drop_coef=(1.0, 1.0, 1.0, 1.0, 1.0))
    #dnn = DeepNeuralNetwork(auto_hid_layer_sz=[1000, 750, 500], 
    #			     auto_nonlin_func=["sigmoid", "sigmoid", "sigmoid"], 
    #			     auto_drop_coef=(1.0, 1.0, 1.0), 
    #			     UnsupervisedModel=RestBolzmanMachine)

    session = tf.InteractiveSession()
    dnn.fit(X_, Y, session=session, optimizer="adam", opt_param_lst=(1e-3, 0.99, 0.999), auto_optimizer="adam", 
            auto_opt_param_lst=(1e-3, 0.99, 0.999), pretrain=True, epochs=10, batch_sz=100, split=True, print_every=20, show_fig=True)





if __name__ == "__main__":
    #deepnetwork()

    explore_single_autoencoder(M2=2200, encoder_type="AutoEncoder", nonlin_func="sigmoid", id=1, auto_optimizer="adam",
                               auto_opt_args=(1e-3, 0.99, 0.999), dropout=1.0, epoch=4, img_lst=[456, 200, 762, 10])
    explore_single_autoencoder(M2=2200, encoder_type="RestBolzmanMachine", nonlin_func="sigmoid", id=1, auto_optimizer="adam",
                               auto_opt_args=(1e-3, 0.99, 0.999), dropout=1.0, epoch=4, img_lst=[456, 200, 762, 10])
    explore_single_autoencoder(M2=(1, 10, 5, 5), encoder_type="ConvolveAutoEncoder", nonlin_func="sigmoid", id=1, auto_optimizer="adam",
                               auto_opt_args=(1e-3, 0.99, 0.999), dropout=1.0, epoch=4, img_lst=[456, 200, 762, 10])

    #explore_data()

    #grid_single_auto_search(M2=(2200, 1800, 1400, 800), nonlin_func=("softmax", "relu", "tanh"), auto_optimizer=("adam", ),
    #                        auto_opt_args=((1e-3, 0.99, 0.999), (1e-3, 0.9, 0.99), (1e-2, 0.99, 0.999), (1e-2, 0.9, 0.99)), dropout=(0.6, ))

    #grid_single_auto_search(M2=(2800, ),
    #                        nonlin_func=("relu", "tanh", ),
    #                        auto_optimizer=("adam",),
    #                        auto_opt_args=((1e-1, 0.99, 0.999), (1e-1, 0.9, 0.99), (1e-2, 0.99, 0.999), (1e-2, 0.9, 0.99)),
    #                        dropout=(1.0, ))