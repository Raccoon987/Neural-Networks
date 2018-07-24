import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import pandas as pd
from itertools import product
import csv
#from ast import literal_eval
#import pickle
from os import walk


def getData(balance_ones=False):
    # images are 48x48 = 2304 size vectors;  N = 35887
    Y, X = [], []
    first = True
    for line in open('C:/Users/Alex/PycharmProjects/Theano_Tensorflow/review_tensorflow/fer2013'):
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



class AutoEncoder():
    def __init__(self, M1, M2, nonlin_func, id, auto_optimizer, auto_opt_args, dropout):
        self.nonlin_func = nonlin_func
        self.id = id
        self.dropout = dropout
        self.build(M1, M2, auto_optimizer, auto_opt_args)

    def set_session(self, session):
        self.session = session

    def nonlinear(self, func):
        if func == "relu":
            return tf.nn.relu
        elif func == "tanh":
            return tf.nn.tanh
        elif func == "softmax":
            return tf.nn.softmax
        elif func == "sigmoid":
            return tf.nn.sigmoid
        elif func == "None":
            return lambda x: x

    def optimizer(self, auto_optimizer, auto_opt_args):
        if auto_optimizer.lower() == "adam":
            optimizer = tf.train.AdamOptimizer
        elif auto_optimizer.lower() == "rmsprop":
            optimizer = tf.train.RMSPropOptimizer
        elif auto_optimizer.lower() == "momentum":
            optimizer = tf.train.MomentumOptimizer
        else:
            raise ValueError('UNSUPPORTED OPTIMIZER TYPE')

        try:
            return optimizer(*auto_opt_args)
        except ValueError:
            print("Uncorrect arguments for " + optimizer + " optimizer")

    def build(self, M1, M2, auto_optimizer, auto_opt_args):
        ''' X ->(W, hidden_bias) -> Z -> (W, output_bias) -> X'  '''
        self.W = tf.Variable(tf.random_normal(shape=(M1, M2)))
        self.hidden_bias = tf.Variable(np.zeros(M2).astype(np.float32))
        self.output_bias = tf.Variable(np.zeros(M1).astype(np.float32))

        self.tfX = tf.placeholder(tf.float32, shape=(None, M1), name="tfX")
        self.Z_hidden = tf.placeholder(tf.float32, shape=(None, M2), name="Z_hidden")
        self.tfX_hat = tf.placeholder(tf.float32, shape=(None, M1), name="Z_hidden")

        self.prob = tf.placeholder_with_default(1.0, shape=())

        logits = self.forward_logits(self.tfX)

        self.cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.tfX, logits=logits))

        ''' "adam": learning_rate, beta1=0.99, beta2=0.999; "rmsprop" : learning_rate, decay=decay, momentum=mu; "momentum" : learning_rate, momentum=mu, use_nesterov=False '''
        # self.train_op = tf.train.AdamOptimizer(learning_rate=1e-1, beta1=0.99, beta2=0.999).minimize(self.cost)
        self.train_op = self.optimizer(auto_optimizer, auto_opt_args).minimize(self.cost)

    def fit(self, X, epochs=1, batch_sz=100, show_fig=True, print_every=20):
        n_batches = X.shape[0] // batch_sz

        print('TRAINING AUTOENCODER WITH NUM: ', self.id)
        costs = []
        self.session.run(tf.global_variables_initializer())
        for i in range(epochs):
            X = shuffle(X)
            for j in range(n_batches):
                Xbatch = X[j*batch_sz:(j*batch_sz + batch_sz)]
                self.session.run(self.train_op, feed_dict={self.tfX: Xbatch, self.prob: self.dropout})
                if j % print_every == 0:
                    costs.append(self.session.run(self.cost, feed_dict={self.tfX: Xbatch, self.prob: self.dropout}))
                    print("epoch: ", i, " batch # ", j, " from ", j, " total batches " , " cost: ",  costs[-1])

        if show_fig:
            plt.plot(costs)
            plt.grid()
            plt.show()

    def forward_hidden(self, X):
        X = tf.nn.dropout(X, self.prob)
        return self.nonlinear(self.nonlin_func)(tf.matmul(X, self.W) + self.hidden_bias)

    def forward_logits(self, X):
        return tf.matmul(self.forward_hidden(X), tf.transpose(self.W)) + self.output_bias

    def forward_output(self, X):
        return tf.nn.sigmoid(self.forward_logits(X))

    def transform(self, X):
        return self.session.run(self.forward_hidden(self.tfX), feed_dict={self.tfX: X})

    def predict(self, X):
        return self.session.run(self.forward_output(self.tfX), feed_dict={self.tfX: X})



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

    def __init__(self, in_fm, out_fm, width, height, nonlin_func, poolsz=(2, 2)):
        # super.__init__(self, M1, M2, an_id, nonlin_func)
        ConvPullLayer.__init__(self, in_fm, out_fm, width, height, nonlin_func, poolsz=[2, 2])


class ConvNeuralNetwork_Image():
    def __init__(self, convpull_layer_sizes, conv_nonlin_functions, poolsz, auto_hid_layer_sz, auto_nonlin_func, auto_drop_coef, hidden_layer_sizes, nonlin_functions, dropout_coef, UnsupervisedModel=AutoEncoder):
        '''length of dropout_coef should == length of hidden_layer + 1:  we make dropout before first layer too'''
        if (len(convpull_layer_sizes) != len(conv_nonlin_functions)) and (len(hidden_layer_sizes) != len(dropout_coef) - 1) and (len(hidden_layer_sizes) != len(nonlin_functions)):
            print("LENGTH OF hidden_layer_sizes PARAMETERS MUST EQUAL TO LENGTH OF dropout_coef - 1 AND EQUAL TO LENGTH OF nonlin_functions")
            raise ValueError
        self.convpull_layer_sizes = convpull_layer_sizes
        self.conv_nonlin_functions = conv_nonlin_functions
        self.poolsz = poolsz
        self.auto_hid_layer_sz = auto_hid_layer_sz
        self.auto_nonlin_func = auto_nonlin_func
        self.auto_drop_coef = auto_drop_coef
        self.unsupervised_model = UnsupervisedModel
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


    def set_session(self, session):
            self.session = session
            for auto_obj in self.auto_hid_lay:
                auto_obj.set_session(session)


    #def fit(self, X, Y, optimizer="adam", optimizer_params=(10e-4, 0.99, 0.999), learning_rate=10e-7, mu=0.99, decay=0.999, reg=10e-3, epochs=400, batch_size=100, split=True, show_fig=False, print_every=20):
    def fit(self, X, Y, session, optimizer="adam", optimizer_params=(10e-4, 0.99, 0.999), auto_optimizer="adam", auto_opt_param_lst=(10e-4, 0.99, 0.999), pretrain=True,
            reg=10e-3, epochs=400, batch_size=100, split=True, show_fig=False, print_every=20, print_tofile=False):
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
            self.convpool_layers.append(ConvPullLayer(input_feature, *outF_wdt_hgt, self.conv_nonlin_functions[index], self.poolsz))
            input_feature = outF_wdt_hgt[0]
        # shape of the image after serie of convolution + maxpool layers
        final_output_width, final_output_height = width / ( self.poolsz[0] ** len(self.convpull_layer_sizes)), height / (self.poolsz[1] ** len(self.convpull_layer_sizes))

        ''' initialize deep network '''
        # size of output feature of last convpull layer * shape of output image
        input_size = int(self.convpull_layer_sizes[-1][0] * final_output_width * final_output_height)
        self.auto_hid_lay = []
        for index, output_size in enumerate(self.auto_hid_layer_sz):
            self.auto_hid_lay.append(self.unsupervised_model(input_size, output_size, self.auto_nonlin_func[index], index, auto_optimizer, auto_opt_param_lst, self.auto_drop_coef[index]))
            input_size = output_size

        self.set_session(session)

        ''' initialize hidden layers '''
        M1 = input_size
        self.hidden_layers = []
        for id in range(len(self.hidden_layer_sizes)):
            '''BEFORE IT WAS HiddenLayerBatchNorm'''
            self.hidden_layers.append(HiddenLayerBatchNorm(M1, self.hidden_layer_sizes[id], id, self.nonlin_functions[id]))
            M1 = self.hidden_layer_sizes[id]

        self.hidden_layers.append(HiddenLayer(M1, K, len(self.hidden_layer_sizes), "None"))
        tfX = tf.placeholder(tf.float32, shape=(None, width, height, color), name="tfX")
        tfT = tf.placeholder(tf.float32, shape=(None, K), name="tfT")
        #self.test = tf.placeholder(tf.float32, shape=(None, D), name="tfTest")
        logits = self.forward(tfX, is_training=True, pretrain=True)

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



    def forward(self, X, is_training, pretrain):
        # tf.nn.dropout automaticaly scales inputs by 1/p_keep
        Z = X
        ''' convpull layer '''
        for c in self.convpool_layers:
            Z = c.layer_forward(Z)
        # since Z is tensorflow tensor object => get_shape(); pass -1 to reshape() when we do not know the exact shape
        ''' switch to fully connected network '''
        Z_shape = Z.get_shape().as_list()
        Z = tf.reshape(Z, [-1, np.prod(Z_shape[1:])])

        ''' deep network '''
        pretrain_epochs = 0
        if pretrain:
            pretrain_epochs = 3

        current = Z
        for auto_enc in self.auto_hid_lay:
            auto_enc.fit(current, epochs=pretrain_epochs, batch_sz=100, show_fig=False, print_every=20)
            current = auto_enc.transform(current)

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




def explore_data():
    X, Y = getData()
    print(X.shape)

def main():
    ''' data shape: (35887, 2304); (35887, 1, 48, 48)'''
    X, Y = getImageData()
    X, Y = X[:5000], Y[:5000]
    # reshape X for tf: N x w x h x c
    X = X.transpose((0, 2, 3, 1))

    model = ConvNeuralNetwork_Image(convpull_layer_sizes=((20, 5, 5), (50, 5, 5)), conv_nonlin_functions=("tanh", "tanh"), poolsz=[2, 2],
                                    hidden_layer_sizes=[500, 300], nonlin_functions=("relu", "relu"), dropout_coef=(0.9, 0.8, 0.8))
    session = tf.InteractiveSession()
    model.set_session(session)
    model.fit(X, Y, optimizer="adam", optimizer_params=(10e-4, 0.99, 0.999), reg=10e-3, epochs=10, batch_size=1000, split=True, show_fig=True, print_every=10, print_tofile=False)

#main()