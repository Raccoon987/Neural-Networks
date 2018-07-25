import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
import pandas as pd
import sys
import os

from sklearn.utils import shuffle
#from Convolution_networks.conv_tf_face_recognition import HiddenLayer, HiddenLayerBatchNorm, ConvPullLayer, getData, getImageData

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
        print("AutoEncoder build method")
        self.W = tf.Variable(tf.random_normal(shape=(M1, M2))* np.sqrt(2.0 / M1))
        self.hidden_bias = tf.Variable(np.zeros(M2).astype(np.float32))
        self.output_bias = tf.Variable(np.zeros(M1).astype(np.float32))

        self.tfX = tf.placeholder(tf.float32, shape=(None, M1), name="tfX")
        self.Z_hidden = tf.placeholder(tf.float32, shape=(None, M2), name="Z_hidden")
        self.tfX_hat = tf.placeholder(tf.float32, shape=(None, M1), name="tfX_reconstructed")

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
                    print("epoch: ", i, " batch # ", j, " from ", n_batches, " total batches " , " cost: ",  costs[-1])

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



class RestBolzmanMachine(AutoEncoder):
    def __init__(self, M1, M2, nonlin_func, id, auto_optimizer, auto_opt_args, dropout=1.0):
        super(RestBolzmanMachine, self).__init__(M1, M2, nonlin_func, id, auto_optimizer, auto_opt_args, dropout=1.0)


    def build(self, M1, M2, auto_optimizer, auto_opt_args):
        ''' X ->(W, hidden_bias) -> Z -> (W, output_bias) -> X'  '''
        print("RestBolzmanMachine build method")
        self.W = tf.Variable(tf.random_normal(shape=(M1, M2)) * np.sqrt(2.0 / M1))
        self.hidden_bias = tf.Variable(np.zeros(M2).astype(np.float32))
        self.output_bias = tf.Variable(np.zeros(M1).astype(np.float32))

        self.prob = tf.placeholder_with_default(1.0, shape=())

        self.tfX = tf.placeholder(tf.float32, shape=(None, M1), name="tfX")

        # conditional probabilities
        self.p_hidden_given_visible = self.nonlinear("sigmoid")(tf.matmul(self.tfX, self.W) + self.hidden_bias)
        # now manually construct Bernoulli distribution: 1 with probability p, 0 with prob. q
        H = tf.to_float(tf.random_uniform(shape = tf.shape(self.p_hidden_given_visible)) < self.p_hidden_given_visible)

        self.p_visible_given_hidden = self.nonlinear("sigmoid")(tf.matmul(H, tf.transpose(self.W)) + self.output_bias)
        X_reconstructed = tf.to_float(tf.random_uniform(shape = tf.shape(self.p_visible_given_hidden)) < self.p_visible_given_hidden)

        objective = tf.reduce_mean(self.free_energy(self.tfX, M1)) - tf.reduce_mean(self.free_energy(X_reconstructed, M1))
        self.train_op = self.optimizer(auto_optimizer, auto_opt_args).minimize(objective)


        ''' not for model optimization - just to observe what happens during training '''
        logits = self.forward_logits(self.tfX)
        self.cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.tfX, logits=logits))


    def free_energy(self, visible, vis_features):
        ''' F(v) = -b.T * visible - sum(log(1 + exp(c(j) + visible.T * W(j)))) '''
        bias = tf.reshape(self.output_bias, (vis_features, 1))
        ''' back to original shape '''
        first_term = tf.reshape(-tf.matmul(visible, bias), (-1, ))

        ''' sum[j](tf.log(1 + tf.exp(tf.matmul(v, self.W) + self.c))) '''
        second_term = -tf.reduce_sum(tf.nn.softplus(tf.matmul(visible, self.W) + self.hidden_bias), axis=1)

        return first_term + second_term


    def fit(self, X, epochs=1, batch_sz=100, show_fig=True, print_every=20):
        n_batches = X.shape[0] // batch_sz

        print('TRAINING RBM_ENCODER WITH NUM: ', self.id)
        costs = []
        self.session.run(tf.global_variables_initializer())
        for i in range(epochs):
            X = shuffle(X)
            for j in range(n_batches):
                Xbatch = X[j * batch_sz:(j * batch_sz + batch_sz)]
                self.session.run(self.train_op, feed_dict={self.tfX: Xbatch})
                if j % print_every == 0:
                    costs.append(self.session.run(self.cost, feed_dict={self.tfX: Xbatch}))
                    print("epoch: ", i, " batch # ", j, " from ", n_batches, " total batches ", " cost: ", costs[-1])

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
        return self.nonlinear(self.nonlin_func)(self.forward_logits(X))

    def transform(self, X):
        return self.session.run(self.p_hidden_given_visible, feed_dict={self.tfX: X})

    def predict(self, X):
        return self.session.run(self.forward_output(self.tfX), feed_dict={self.tfX: X})
    ''' ??? '''
    #def predict(self, X):
    #    return self.session.run(self.forward_output(self.tfX), feed_dict={self.tfX: X})



class ConvolveAutoEncoder():
    def __init__(self, in_fm, out_fm, width, height, nonlin_func, id, auto_optimizer, auto_opt_args, poolsz=[2, 2]):
        """ sz = (width, height, in_fm, out_fm);   in_fm - input feature map size, out_fm - output feature map size """
        self.id = id
        self.poolsz = poolsz
        self.nonlin_func = nonlin_func
        """ convolution filter size """
        self.params = [in_fm, out_fm, width, height]
        self.build((width, height, in_fm, out_fm), poolsz, auto_optimizer, auto_opt_args)

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

    # filter shapes are expected to be: filter width x filter height x input feature maps x output feature maps
    # ??? sqrt( height*weight*color + feature*heiht*weight/(2*2)) ??? why so?
    def init_filter(self, shape, poolsz):
        w = np.random.randn(*shape) * np.sqrt(2) / np.sqrt(np.prod(shape[:-1]) + shape[-1] * np.prod(shape[:-2] / np.prod(poolsz)))
        return w.astype(np.float32), np.zeros(shape[-1], dtype=np.float32)

    def build(self, sz, poolsz, auto_optimizer, auto_opt_args):
        ''' X ->(W, hidden_bias) -> Z -> (W, output_bias) -> X'  '''
        self.W, self.hidden_bias = map(tf.Variable, self.init_filter(sz, poolsz))
        self.output_bias = tf.Variable(np.zeros(sz[-2]).astype(np.float32))

        self.tfX = tf.placeholder(tf.float32, shape=(None, 48, 48, sz[2]), name="tfX")
        self.Z_hidden = tf.placeholder(tf.float32, shape=(None, 24, 24, sz[1]), name="Z_hidden")
        self.tfX_hat = tf.placeholder(tf.float32, shape=(None, 48, 48, sz[2]), name="tfX_reconstructed")

        self.tfX_predict = self.forward_output(self.tfX)

        logits = self.forward_logits(self.tfX)

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.tfX, logits=logits))

        ''' "adam": learning_rate, beta1=0.99, beta2=0.999; "rmsprop" : learning_rate, decay=decay, momentum=mu; "momentum" : learning_rate, momentum=mu, use_nesterov=False '''
        self.train_op = self.optimizer(auto_optimizer, auto_opt_args).minimize(self.cost)

    def convol(self, X):
        return tf.nn.bias_add(tf.nn.conv2d(X, self.W, strides=[1, 1, 1, 1], padding="SAME"), self.hidden_bias)

    def deconvol(self, Z_unpool):
        return tf.contrib.layers.conv2d_transpose(Z_unpool, num_outputs=self.params[0], kernel_size=self.params[2:], stride=[1, 1], padding='SAME', activation_fn=None)
                                                 #activation_fn=self.nonlinear(self.nonlin_func))

    def pool(self, X_conv):
        return tf.nn.max_pool(X_conv, ksize=([1] + self.poolsz + [1]), strides=([1] + self.poolsz + [1]), padding="SAME")

    def unpool(self, Z, factor):
        size = [int(Z.shape[1] * factor[0]), int(Z.shape[2] * factor[1])]
        return tf.image.resize_bilinear(Z, size=size, align_corners=None, name=None)

    def fit(self, X, epochs=1, batch_sz=100, show_fig=True, print_every=20):
        n_batches = X.shape[0] // batch_sz

        print('TRAINING CONVOLUTIONAL AUTOENCODER WITH NUM: ', self.id)
        costs = []
        self.session.run(tf.global_variables_initializer())
        for i in range(epochs):
            X = shuffle(X)
            for j in range(n_batches):
                Xbatch = X[j*batch_sz:(j*batch_sz + batch_sz)]
                self.session.run(self.train_op, feed_dict={self.tfX: Xbatch})
                if j % print_every == 0:
                    costs.append(self.session.run(self.cost, feed_dict={self.tfX: Xbatch}))
                    print("epoch: ", i, " batch # ", j, " from ", n_batches, " total batches " , " cost: ",  costs[-1])

        if show_fig:
            plt.plot(costs)
            plt.grid()
            plt.show()

    def forward_hidden(self, X):
        conv = self.convol(X)
        pool = self.pool(conv)

        return self.nonlinear(self.nonlin_func)(pool)

    def forward_logits(self, X):
        un_pool = self.unpool(self.forward_hidden(X), self.poolsz)
        back_convolve = self.deconvol(un_pool)
        return back_convolve

    def forward_output(self, X):
        return self.nonlinear(self.nonlin_func)(self.forward_logits(X))

    def predict(self, X):
        return self.session.run(self.tfX_predict, feed_dict={self.tfX: X})










class DeepNeuralNetwork():
    def __init__(self, auto_hid_layer_sz, auto_nonlin_func, auto_drop_coef, UnsupervisedModel=AutoEncoder):
        if (len(auto_hid_layer_sz) != len(auto_nonlin_func)):
            print("SIZE OF THE LAYERS LIST MUST BE EQUAL WITH SIZE OF NONLINEAR FUNCTION LIST ")
            raise ValueError

        self.auto_hid_layer_sz = auto_hid_layer_sz
        self.auto_nonlin_func = auto_nonlin_func
        self.auto_drop_coef = auto_drop_coef
        self.unsupervised_model = UnsupervisedModel
        self.auto_hid_lay = []

    def choose_optimizer(self, optimizer, opt_args):
        if optimizer.lower() == "adam":
            optimizer = tf.train.AdamOptimizer
        elif optimizer.lower() == "rmsprop":
            optimizer = tf.train.RMSPropOptimizer
        elif optimizer.lower() == "momentum":
            optimizer = tf.train.MomentumOptimizer
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

    def fit(self, X, Y, session, optimizer, opt_param_lst, auto_optimizer, auto_opt_param_lst, pretrain=True, epochs=1, batch_sz=100, split=True, print_every=20, show_fig=False):
        K = len(set(Y))
        if isinstance(X, np.ndarray):
            X = X.astype(np.float32)
        Y = y_hot_encoding(Y).astype(np.float32)
        X, Y = shuffle(X, Y)
        if split:
            X, Y = X[:-1000], Y[:-1000]
            Xvalid, Yvalid = X[-1000:], Y[-1000:]
        else:
            Xvalid, Yvalid = X, Y
        Yvalid_flat = np.argmax(Yvalid, axis=1)

        input_size = X.shape[1]
        for index, output_size in enumerate(self.auto_hid_layer_sz):
            self.auto_hid_lay.append(self.unsupervised_model(input_size, output_size, self.auto_nonlin_func[index], index, auto_optimizer, auto_opt_param_lst, self.auto_drop_coef[index]))
            input_size = output_size

        self.set_session(session)

        """ build last regression layer """
        self.W = tf.Variable(tf.random_normal(shape=(self.auto_hid_layer_sz[-1], K)))
        self.b = tf.Variable(np.zeros(K).astype(np.float32))
        tfX = tf.placeholder(tf.float32, shape=(None, X.shape[1]), name="tfX")
        tfT = tf.placeholder(tf.float32, shape=(None, K), name="tfT")

        logits = self.forward(tfX)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tfT))
        train_op = self.choose_optimizer(optimizer=optimizer, opt_args=opt_param_lst).minimize(cost)
        #train_op = tf.train.AdamOptimizer(1e-2).minimize(cost)
        prediction = tf.argmax(logits, axis=1)

        pretrain_epochs = 0
        if pretrain:
            pretrain_epochs = 8

        current = X
        for auto_enc in self.auto_hid_lay:
            auto_enc.fit(current, epochs=pretrain_epochs, batch_sz=100, show_fig=False, print_every=20)
            current = auto_enc.transform(current)

        n_batches = X.shape[0] // batch_sz
        print('TRAINING DEEP NEURAL NETWORK')
        costs = []
        self.session.run(tf.global_variables_initializer())
        for i in range(epochs):
            X, Y = shuffle(X, Y)
            for j in range(n_batches):
                Xbatch = X[j*batch_sz:(j*batch_sz + batch_sz)]
                Ybatch = Y[j*batch_sz:(j*batch_sz + batch_sz)]
                self.session.run(train_op, feed_dict={tfX: Xbatch, tfT: Ybatch})

                if j % print_every == 0:
                    costs.append(self.session.run(cost, feed_dict={tfX: Xvalid, tfT: Yvalid}))
                    c = self.session.run(prediction, feed_dict={tfX: Xvalid, tfT: Yvalid})
                    print("i:", i, "j:", j, "nb:", n_batches, "cost:", costs[-1], "error_rate:", error_rate(Yvalid_flat, c))

        if show_fig:
            plt.plot(costs)
            plt.grid()
            plt.show()

    def forward(self, X):
        current = X
        for auto_layer in self.auto_hid_lay:
            current = auto_layer.forward_hidden(current)

        #last logistic layer: RETURN LOGITS
        return tf.matmul(current, self.W) + self.b