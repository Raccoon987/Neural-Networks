import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import pandas as pd
from itertools import product
import csv
from preprocess import getData, getImageData, init_weight_and_bias, init_filter, y_hot_encoding, error_rate
#import pickle
from os import walk



class HiddenLayer():
    def __init__(self, M1, M2, an_id, nonlin_func):
        self.id = an_id
        self.M1 = M1
        self.M2 = M2
        '''self.params contain W and b for particular layer'''
        self.params = list(map(tf.Variable, init_weight_and_bias(M1, M2)))
        self.nonlin_func = nonlin_func

    def layer_forward(self, X):
        return self.nonlinear(self.nonlin_func)(tf.matmul(X, self.params[0]) + self.params[1])

    def nonlinear(self, func):
        if func == "relu":
            return tf.nn.relu
        elif func == "tanh":
            return  tf.nn.tanh
        elif func == "softmax":
            return tf.nn.softmax
        elif func == "None":
            return lambda x:x


class HiddenLayerBatchNorm(HiddenLayer):
    """or not to write __init__ method at all"""
    def __init__(self, M1, M2, an_id, nonlin_func):
        #super.__init__(self, M1, M2, an_id, nonlin_func)
        HiddenLayer.__init__(self, M1, M2, an_id, nonlin_func)

        self.gamma = tf.Variable(np.ones(M2).astype(np.float32))
        self.beta = tf.Variable(np.zeros(M2).astype(np.float32))

        #for test time
        self.running_mean = tf.Variable(np.zeros(M2).astype(np.float32), trainable=False)
        self.running_var = tf.Variable(np.zeros(M2).astype(np.float32), trainable=False)

    def layer_forward(self, X, is_training, decay=0.9):
        linear = tf.matmul(X, self.params[0])

        if is_training:
            batch_mean, batch_var = tf.nn.moments(linear, axes=[0])
            update_running_mean = tf.assign(self.running_mean, self.running_mean * decay + batch_mean * (1 - decay))
            update_running_var = tf.assign(self.running_var, self.running_var * decay + batch_var * (1 - decay))

            with tf.control_dependencies([update_running_mean, update_running_var]):
                normalized_linear = tf.nn.batch_normalization(linear, batch_mean, batch_var, self.beta, self.gamma, variance_epsilon=1e-4)
        else:
            normalized_linear = tf.nn.batch_normalization(linear, self.running_mean, self.running_var, self.beta, self.gamma, variance_epsilon=1e-4)

        return self.nonlinear(self.nonlin_func)(normalized_linear)


class ConvPullLayer():
    def __init__(self, in_fm, out_fm, width, height, nonlin_func, poolsz=[2, 2]):
        # mi = input feature map size; mo = output feature map size
        sz = (width, height, in_fm, out_fm)
        ''' self.params contains W and b - convolution weights and bias'''
        self.params = list(map(tf.Variable, init_filter(sz, poolsz)))
        self.poolsz = poolsz
        self.nonlin_func = nonlin_func

    def layer_forward(self, X):
        conv_out = tf.nn.bias_add(tf.nn.conv2d(X, self.params[0], strides=[1, 1, 1, 1], padding="SAME"), self.params[1])
        pool_out = tf.nn.max_pool(conv_out, ksize=([1] + self.poolsz + [1]), strides=([1] + self.poolsz + [1]), padding="SAME")
        return self.nonlinear(self.nonlin_func)(pool_out)


    def nonlinear(self, func):
        if func == "relu":
            return tf.nn.relu
        elif func == "tanh":
            return tf.nn.tanh
        elif func == "softmax":
            return tf.nn.softmax
        elif func == "None":
            return lambda x: x

class ConvPullLayerBatchNorm(ConvPullLayer):
    """or not to write __init__ method at all"""
    """ not yet realized """

    def __init__(self, in_fm, out_fm, width, height, nonlin_func, poolsz=(2, 2)):
        # super.__init__(self, M1, M2, an_id, nonlin_func)
        ConvPullLayer.__init__(self, in_fm, out_fm, width, height, nonlin_func, poolsz=[2, 2])


class ConvNeuralNetwork_Image():
    def __init__(self, convpull_layer_sizes, conv_nonlin_functions, poolsz, hidden_layer_sizes, nonlin_functions, dropout_coef):
        '''length of dropout_coef should == length of hidden_layer + 1:  we make dropout before first layer too'''
        if (len(convpull_layer_sizes) != len(conv_nonlin_functions)) and \
           (len(hidden_layer_sizes) != len(dropout_coef) - 1) and \
           (len(hidden_layer_sizes) != len(nonlin_functions)):
            
            print("LENGTH OF hidden_layer_sizes PARAMETERS MUST EQUAL TO LENGTH OF dropout_coef - 1 AND EQUAL TO LENGTH OF nonlin_functions")
            raise ValueError
        self.convpull_layer_sizes = convpull_layer_sizes
        self.conv_nonlin_functions = conv_nonlin_functions
        self.poolsz = poolsz
        self.hidden_layer_sizes = hidden_layer_sizes
        self.nonlin_functions = nonlin_functions
        self.dropout = dropout_coef
        '''counter for crossValidation - to save session name'''

    def set_session(self, session):
        self.session = session

    def optimizer(self, optimizer, opt_args):
        if optimizer.lower() == "adam":
            optimizer = tf.train.AdamOptimizer
        elif optimizer.lower() == "rmsprop":
            optimizer = tf.train.RMSPropOptimizer
        elif optimizer.lower() == "momentum":
            optimizer = tf.train.MomentumOptimizer
        elif optimizer.lower() == "proximal":
            optimizer = tf.train.ProximalGradientDescentOptimizer
        else:
            raise ValueError('UNSUPPORTED OPTIMIZER TYPE')

        try:
            return optimizer(*opt_args)
        except ValueError:
            print("Uncorrect arguments for " + optimizer + " optimizer")

    #def fit(self, X, Y, optimizer="adam", optimizer_params=(10e-4, 0.99, 0.999), learning_rate=10e-7, mu=0.99, 
    #        decay=0.999, reg=10e-3, epochs=400, batch_size=100, split=True, show_fig=False, print_every=20):
    
    def fit(self, X, Y, optimizer="adam", optimizer_params=(10e-4, 0.99, 0.999), reg=10e-3, epochs=400, 
            batch_size=100, split=True, show_fig=False, print_every=20, print_tofile=False):
        
	#learning_rate, mu, decay, reg = map(np.float32, [learning_rate, mu, decay, reg])
        K = len(set(Y))
        X, Y = X.astype(np.float32), y_hot_encoding(Y).astype(np.float32)
        X, Y = shuffle(X, Y)
        if split:
            Xvalid, Yvalid = X[-1000:], Y[-1000:]
            X, Y = X[:-1000], Y[:-1000]
        else:
            Xvalid, Yvalid = X, Y
        Yvalid_flat = np.argmax(Yvalid, axis=1)

        ''' initialize convpool layers '''
        N, width, height, color = X.shape
        input_feature = color
        self.convpool_layers = []
        # in self.convpull_layer_sizes should be (new_feature, filter_width, filter_height)
        for index, outF_wdt_hgt in enumerate(self.convpull_layer_sizes):
            self.convpool_layers.append(ConvPullLayer(input_feature, 
						      *outF_wdt_hgt, 
						      self.conv_nonlin_functions[index], 
						      self.poolsz))
            input_feature = outF_wdt_hgt[0]
        
	# shape of the image after serie of convolution + maxpool layers
        final_output_width, final_output_height = width / ( self.poolsz[0] ** len(self.convpull_layer_sizes)), \
						  height / (self.poolsz[1] ** len(self.convpull_layer_sizes))

        ''' initialize hidden layers '''
        # size of output feature of last convpull layer * shape of output image
        M1 = int(self.convpull_layer_sizes[-1][0] * final_output_width * final_output_height)
        self.hidden_layers = []
        for id in range(len(self.hidden_layer_sizes)):
            '''BEFORE IT WAS HiddenLayerBatchNorm'''
            self.hidden_layers.append(HiddenLayerBatchNorm(M1, self.hidden_layer_sizes[id], id, self.nonlin_functions[id]))
            M1 = self.hidden_layer_sizes[id]

        self.hidden_layers.append(HiddenLayer(M1, K, len(self.hidden_layer_sizes), "None"))
        tfX = tf.placeholder(tf.float32, shape=(None, width, height, color), name="tfX")
        tfT = tf.placeholder(tf.float32, shape=(None, K), name="tfT")
        #self.test = tf.placeholder(tf.float32, shape=(None, D), name="tfTest")
        logits = self.forward(tfX, is_training=True)

        rcost = reg * sum([tf.nn.l2_loss(coefs) for layer in (self.convpool_layers + self.hidden_layers) for coefs in layer.params])
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tfT)) + rcost
        prediction = self.predict(tfX)

        train_op = self.optimizer(optimizer=optimizer, opt_args=optimizer_params).minimize(cost)

        n_batches = int(N / batch_size)
        batch_costs = []
        valid_costs = []
        error = []

        self.session.run(tf.global_variables_initializer())

        for i in range(epochs):
            X, Y = shuffle(X, Y)
            for j in range(n_batches):
                Xbatch = X[j*batch_size:(j*batch_size + batch_size)]
                Ybatch = Y[j*batch_size:(j*batch_size + batch_size)]
                self.session.run(train_op, feed_dict={tfX: Xbatch, tfT: Ybatch})
                if j % print_every == 0:
                    batch_costs.append(self.session.run(cost, feed_dict={tfX: Xbatch, tfT: Ybatch}))
                    valid_costs.append(self.session.run(cost, feed_dict={tfX: Xvalid, tfT: Yvalid}))
                    p = self.session.run(prediction, feed_dict={tfX: Xvalid, tfT: Yvalid})
                    err_rate = error_rate(Yvalid_flat, p)
                    error.append(err_rate)
                    print("i:", i, "j:", j, "nb:", n_batches, "cost:", valid_costs[-1], "error_rate:", err_rate)

        print("Done!")

        if show_fig:
            plt.plot(valid_costs)
            plt.xlabel('20 * iteration', fontsize=14)
            plt.ylabel('cost', fontsize=14)
            plt.grid()
            plt.show()

        if print_tofile:
            my_df = pd.DataFrame([batch_costs, valid_costs, error])
            my_df.to_csv(print_tofile, index=False, header=False)



    def forward(self, X, is_training):
        # tf.nn.dropout automaticaly scales inputs by 1/p_keep
        Z = X
        ''' convpull layer '''
        for c in self.convpool_layers:
            Z = c.layer_forward(Z)
        # since Z is tensorflow tensor object => get_shape(); pass -1 to reshape() when we do not know the exact shape
        ''' switch to fully connected network '''
        Z_shape = Z.get_shape().as_list()
        Z = tf.reshape(Z, [-1, np.prod(Z_shape[1:])])

        ''' vanilla network with dropout '''
        if is_training:
            Z = tf.nn.dropout(Z, self.dropout[0])
        for h, p in zip(self.hidden_layers[:-1], self.dropout[1:]):
            Z = h.layer_forward(Z, is_training)
            if is_training:
                Z = tf.nn.dropout(Z, p)
        return self.hidden_layers[-1].layer_forward(Z)

    def predict(self, X):
        act = self.forward(X, is_training=False)
        return tf.argmax(act, 1)

    def make_prediction(self, X):
        if not isinstance(X, np.ndarray):
            X = X.astype(np.float32).toarray()

        prediction = self.predict(self.test)
        return self.session.run(prediction, feed_dict={self.test: X})

    def score(self, X, Y):
        #if not isinstance(Y, np.ndarray):
        Y = y_hot_encoding(Y).astype(np.float32)
        p = self.make_prediction(X)
        return 1 - error_rate(np.argmax(Y, axis=1), p)

