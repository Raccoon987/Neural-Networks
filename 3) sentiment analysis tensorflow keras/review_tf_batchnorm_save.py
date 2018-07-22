import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from preprocess import get_data, y_hot_encoding, error_rate, init_weight_and_bias
from sklearn.utils import shuffle

from preprocess import vectorizer, crossValidation, tokenize, stem_tokens
from itertools import product



"""SECOND VERSION WITH SAVING SESSION"""
class HiddenLayer_1():
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


class HiddenLayerBatchNorm(HiddenLayer_1):
    """or not to write __init__ method at all"""
    def __init__(self, M1, M2, an_id, nonlin_func):
        #super.__init__(self, M1, M2, an_id, nonlin_func)
        HiddenLayer_1.__init__(self, M1, M2, an_id, nonlin_func)

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



class NeuralNetworkBatchN():
    def __init__(self, hidden_layer_sizes, nonlin_functions, dropout_coef):
        '''length of dropout_coef should == length of hidden_layer + 1:  we make dropout before first layer too'''
        if (len(hidden_layer_sizes) != len(dropout_coef) - 1) and (len(hidden_layer_sizes) != len(nonlin_functions)):
            print("LENGTH OF hidden_layer_sizes PARAMETERS MUST EQUAL TO LENGTH OF dropout_coef - 1 AND EQUAL TO LENGTH OF nonlin_functions")
            raise ValueError
        self.hidden_layer_sizes = hidden_layer_sizes
        self.nonlin_functions = nonlin_functions
        self.dropout = dropout_coef
        '''counter for crossValidation - to save session name'''
        self.counter = 0

    def fit(self, X, Y, learning_rate=10e-7, mu=0.99, decay=0.999, reg=10e-3, epochs=400, batch_size=100, split=True, show_fig=False, print_every=20):
        self.epochs = epochs
        K = len(set(Y))
        X, Y = X.astype(np.float32).toarray(), y_hot_encoding(Y).astype(np.float32)
        X, Y = shuffle(X, Y)
        if split:
            Xvalid, Yvalid = X[-1000:], Y[-1000:]
            X, Y = X[:-1000], Y[:-1000]
        else:
            Xvalid, Yvalid = X, Y
        Yvalid_flat = np.argmax(Yvalid, axis=1)

        '''Clears the default graph stack and resets the global default graph.'''
        tf.reset_default_graph()

        '''initialize hidden layers'''
        N, D = X.shape
        M1 = D
        self.hidden_layers = []
        for id in range(len(self.hidden_layer_sizes)):
            self.hidden_layers.append(HiddenLayerBatchNorm(M1, self.hidden_layer_sizes[id], id, self.nonlin_functions[id]))
            M1 = self.hidden_layer_sizes[id]
        self.hidden_layers.append(HiddenLayer_1(M1, K, len(self.hidden_layer_sizes), "None"))

        tfX = tf.placeholder(tf.float32, shape=(None, D), name="tfX")
        tfT = tf.placeholder(tf.float32, shape=(None, K), name="tfT")
        logits = self.forward(tfX, is_training=True)

        rcost = reg * sum([tf.nn.l2_loss(coefs) for layer in self.hidden_layers for coefs in layer.params])
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tfT)) + rcost
        prediction = self.predict(tfX)

        #train_op = tf.train.RMSPropOptimizer(learning_rate, decay=decay, momentum=mu).minimize(cost)
        train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.99, beta2=0.999).minimize(cost)
        #train_op = tf.train.MomentumOptimizer(learning_rate, momentum=mu, use_nesterov=False).minimize(cost)
        #train_op = tf.train.ProximalGradientDescentOptimizer(learning_rate, l2_regularization_strength=0.0, use_locking=False).minimize(cost)

        n_batches = int(N / batch_size)
        costs = []
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            for i in range(epochs):
                X, Y = shuffle(X, Y)
                for j in range(n_batches):
                    Xbatch = X[j*batch_size:(j*batch_size + batch_size)]
                    Ybatch = Y[j*batch_size:(j*batch_size + batch_size)]
                    session.run(train_op, feed_dict={tfX: Xbatch, tfT: Ybatch})

                    if j % print_every == 0:
                        costs.append(session.run(cost, feed_dict={tfX: Xvalid, tfT: Yvalid}))
                        p = session.run(prediction, feed_dict={tfX: Xvalid, tfT: Yvalid})
                        print("i:", i, "j:", j, "nb:", n_batches, "cost:", costs[-1], "error_rate:", error_rate(Yvalid_flat, p))
            saver = tf.train.Saver()
            '''Now, save the graph'''
            saver.save(session, './my_model-' + str(self.counter), global_step=self.epochs)
            print("Done!")

        if show_fig:
            plt.plot(costs)
            plt.show()

    def forward(self, X, is_training):
        # tf.nn.dropout automaticaly scales inputs by 1/p_keep
        Z = X
        if is_training:
            Z = tf.nn.dropout(X, self.dropout[0])

        for h, p in zip(self.hidden_layers[:-1], self.dropout[1:]):
            Z = h.layer_forward(Z, is_training)
            if is_training:
                Z = tf.nn.dropout(Z, p)
        return  self.hidden_layers[-1].layer_forward(Z)

    def predict(self, X):
        act = self.forward(X, is_training=False)
        return tf.argmax(act, 1)

    def make_prediction(self, X):
        if not isinstance(X, np.ndarray):
            X = X.astype(np.float32).toarray()
        with tf.Session() as sess:
            '''First let's load meta graph and restore weights'''
            saver = tf.train.import_meta_graph('my_model-' + str(self.counter) + "-" + str(self.epochs) + '.meta')
            saver.restore(sess, tf.train.latest_checkpoint('./'))
            self.counter += 1

            graph = tf.get_default_graph()
            tfX = graph.get_tensor_by_name("tfX:0")
            prediction = self.predict(tfX)
            return sess.run(prediction, feed_dict={tfX: X})

    def score(self, X, Y):
        #if not isinstance(Y, np.ndarray):
        Y = y_hot_encoding(Y).astype(np.float32)
        p = self.make_prediction(X)
        return 1 - error_rate(np.argmax(Y, axis=1), p)