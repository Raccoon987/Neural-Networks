
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
import string
import itertools
import datetime

from sklearn.utils import shuffle
from tensorflow.contrib.rnn import static_rnn as get_rnn_output
from tensorflow.contrib.rnn import GRUCell, BasicRNNCell, LSTMCell   # BasicLSTMCell

f_path = lambda f_name: os.path.realpath(os.path.join(os.getcwd(), f_name)).replace('\\', '/')

def init_weight_and_bias(M1, M2):
    #return (np.random.randn(M1, M2) * np.sqrt(2.0 / M1)).astype(np.float32), (np.zeros(M2)).astype(np.float32)
    return (np.random.randn(M1, M2) / np.sqrt(M2 + M1)).astype(np.float32), (np.zeros(M2)).astype(np.float32)


def get_pos_data(train_filename, test_filename):
    try:
        data_train = pd.read_csv(train_filename, sep=" ", header=None, names = ["word", "pos", "smth"])
        data_test = pd.read_csv(test_filename, sep=" ", header=None, names = ["word", "pos", "smth"])
        word2idx, tag2idx  = {}, {}

        data_train.replace(["?", "!"], ".", inplace=True)
        data_test.replace(["?", "!"], ".", inplace=True)

        for v in data_train["word"].unique():
            if v not in string.punctuation:
                word2idx[v] = word2idx.get(v, 1) + len(word2idx)
        for p in data_train["pos"].unique():
            if (p not in string.punctuation) and (p != "''") and (p != "``"):
                tag2idx[p] = tag2idx.get(p, 1) + len(tag2idx)

        # ['this','is','a','cat','.','hello'] => [['this','is','a','cat'], ['hello']] => [[id_1,id_2,id_3,id_4], [id_5]]
        def fun(dict2idx, lst):
            return [list(map(lambda x: dict2idx.get(x, 0), list(g))) for k, g in itertools.groupby(lst, lambda x: x == '.') if not k]

        X_train, Y_train = fun(word2idx, list(data_train["word"])), fun(tag2idx, list(data_train["pos"]))
        X_test, Y_test = fun(word2idx, list(data_test["word"])), fun(tag2idx, list(data_test["pos"]))

        return X_train, Y_train, X_test, Y_test, word2idx, tag2idx

    except IOError:
        print("file not exist")





class RecurrentPOSClass():
    '''  D - embedding size; V - vocabulary size; K - output classes '''
    def __init__(self, D, hid_lay_sizes):
        self.D = D
        self.hid_lay_sizes = hid_lay_sizes

    def nonlinear(self, func):
        nonlin_dict = {"relu": tf.nn.relu,
                       "tanh": tf.nn.tanh,
                       "softmax": tf.nn.softmax,
                       "sigmoid": tf.nn.sigmoid,
                       "None": lambda x: x}
        if func in nonlin_dict.keys():
            return nonlin_dict[func]

    def set_session(self, session):
        self.session = session

    def optimizer(self, auto_optimizer, auto_opt_args):
        optimizer_dict = {"graddes": tf.train.GradientDescentOptimizer,
                          "adadelta": tf.train.AdadeltaOptimizer,
                          "adagrad": tf.train.AdagradOptimizer,
                          "adagradD": tf.train.AdagradDAOptimizer,
                          "momentum": tf.train.MomentumOptimizer,
                          "adam": tf.train.AdamOptimizer,
                          "ftlr": tf.train.FtrlOptimizer,
                          "proxgrad": tf.train.ProximalGradientDescentOptimizer,
                          "proxadagrad": tf.train.ProximalAdagradOptimizer,
                          "rms": tf.train.RMSPropOptimizer}

        if auto_optimizer.lower() in optimizer_dict.keys():
            optimizer = optimizer_dict[auto_optimizer.lower()]
        else:
            raise ValueError('UNSUPPORTED OPTIMIZER TYPE')

        try:
            return optimizer(*auto_opt_args)
        except ValueError:
            print("Uncorrect arguments for " + optimizer + " optimizer")


    def helper(self, dim):
        return list(map(tf.Variable, init_weight_and_bias(dim[0], dim[1])))


    def model_initializer(self, V, K, sq_length, recurr_unit, nonlin_func, optimizer, optimizer_args, reg):

        self.input_words = tf.placeholder(tf.int32, shape=(None, sq_length), name="tfX")
        self.target_POS = tf.placeholder(tf.int32, shape=(None, sq_length), name="tfT")
        num_samples = tf.shape(self.input_words)[0]

        self.hidden_layers = []
        M_input = self.D

        self.W_embed = self.helper((V, self.D))
        self.W_out = self.helper((self.hid_lay_sizes[-1], K))

        Xw = tf.nn.embedding_lookup(self.W_embed[0], self.input_words)
        # converts x from a tensor of shape N x T x M into a list of length T, where each element is a tensor of shape N x M
        Xw = tf.unstack(Xw, sq_length, 1)

        output = Xw
        for idx, layer_sz in enumerate(self.hid_lay_sizes):
            rnn_unit = recurr_unit[idx](num_units=layer_sz, activation=self.nonlinear(nonlin_func[idx]))
            output, _ = get_rnn_output(rnn_unit, output, dtype=tf.float32)

        # outputs are now of size (T, N, M) => make it (N, T, M); M - is last hidden layer size
        output = tf.transpose(output, (1, 0, 2))
        output = tf.reshape(output, (sq_length * num_samples, self.hid_lay_sizes[-1]))  # NT x M


        logits = tf.matmul(output, self.W_out[0]) + self.W_out[1]  # NT x K
        self.prediction = tf.reshape(tf.argmax(logits, axis=1), (num_samples, sq_length))
        #self.out_prob = tf.nn.softmax(logits)

        l2_loss = reg * sum(
                             tf.nn.l2_loss(tf_var)
                             for tf_var in tf.trainable_variables()
                             if not ("noreg" in tf_var.name or "Bias" in tf_var.name)
                            )
        ''' tf.reduce_sum([beta*tf.nn.l2_loss(var) for var in tf.trainable_variables()]) '''

        self.cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits,
                labels=tf.reshape(self.target_POS, [-1])
            )
        ) + l2_loss

        self.train_op = self.optimizer(optimizer, optimizer_args).minimize(self.cost)


    def fit(self, vocab_size, X_train, Y_train, X_test, Y_test, recurr_unit, nonlin_func, optimizer, optimizer_args, reg, batch_sz, epochs):

        ''' V - vocabulary size; K - output classes  '''
        V = vocab_size + 1
        K = len(set(itertools.chain(*Y_train)) | set(itertools.chain(*Y_test)))  # num classes

        sequence_length = max(len(x) for x in X_train + X_test)
        pad_Sq = lambda data: tf.keras.preprocessing.sequence.pad_sequences(data, maxlen=sequence_length)
        X_train, Y_train, X_test, Y_test = pad_Sq(X_train), pad_Sq(Y_train), pad_Sq(X_test), pad_Sq(Y_test)

        self.model_initializer(V, K, sequence_length, recurr_unit, nonlin_func, optimizer, optimizer_args, reg)
        self.session.run(tf.global_variables_initializer())

        train_n_batches, test_n_batches = len(X_train) // batch_sz, len(X_test) // batch_sz
        costs = [0] * epochs
        train_accuracy, test_accuracy = [0] * epochs, [0] * epochs

        for epoch in range(epochs):
            t0 = datetime.datetime.now()
            X_train, Y_train = shuffle(X_train, Y_train)

            n_total, n_correct = 0, 0
            for batch_n in range(train_n_batches):
                ''' sequenceLengths is needed to find where ends previous sentence and starts new one  '''
                X, Y = X_train[batch_n * batch_sz:(batch_n + 1) * batch_sz], Y_train[batch_n * batch_sz:(batch_n + 1) * batch_sz]


                self.session.run(self.train_op, feed_dict={self.input_words: X, self.target_POS: Y})

                c, predict = self.session.run([self.cost, self.prediction], feed_dict={self.input_words: X, self.target_POS: Y})

                costs[epoch] += c

                ''' accuracy '''
                for y_i, p_i in zip(Y, predict):
                  # we don't care about the padded entries so ignore them
                  n_correct += np.sum(y_i[y_i > 0] == p_i[y_i > 0])
                  n_total += len(y_i[y_i > 0])


                if batch_n % 50 == 0:
                    sys.stdout.write("j/N: %d/%d/r" % (batch_n, train_n_batches))
                    sys.stdout.write('\n')
                    sys.stdout.flush()

            train_accuracy[epoch] = float(n_correct)/n_total

            n_total, n_correct = 0, 0
            for batch_n in range(test_n_batches):
                ''' sequenceLengths is needed to find where ends previous sentence and starts new one  '''
                X, Y = X_test[batch_n * batch_sz:(batch_n + 1) * batch_sz], Y_test[batch_n * batch_sz:(batch_n + 1) * batch_sz]

                predict = self.session.run(self.prediction, feed_dict={self.input_words: X, self.target_POS: Y})

                ''' accuracy '''
                for y_i, p_i in zip(Y, predict):
                  # we don't care about the padded entries so ignore them
                  n_correct += np.sum(y_i[y_i > 0] == p_i[y_i > 0])
                  n_total += len(y_i[y_i > 0])

            test_accuracy[epoch] = float(n_correct) / n_total

            print("epoch: ", epoch, \
                  " cost: ", costs[epoch], \
                  "train dataset correct rate: ", train_accuracy[epoch], \
                  "test dataset correct rate: ", test_accuracy[epoch], \
                  " time for epoch: ", (datetime.datetime.now() - t0))

        fig, ax1 = plt.subplots()
        ax1.plot(np.log(costs), 'b-')
        ax1.set_xlabel('epochs')
        # Make the y-axis label, ticks and tick labels match the line color.
        ax1.set_ylabel('log cost', color='b')

        ax2 = ax1.twinx()
        ax2.plot(train_accuracy, 'r-')
        ax2.set_ylabel('train_accuracy', color='r')
        ax2.plot(test_accuracy, 'g-')


        fig.tight_layout()
        ax1.set_title('Log cost Curve' + "train_acc: " + str(round(train_accuracy[-1], 3)) + \
                      "test_acc: " + str(round(test_accuracy[-1], 3)) )
        ax1.grid()
        plt.show()





if __name__ == "__main__":
    print(tf.__version__)
    X_train, Y_train, X_test, Y_test, word2idx, tag2idx = get_pos_data(f_path('POS_train.txt'), f_path('POS_test.txt'))

    ''' GRUCell, BasicRNNCell, LSTMCell '''
    print("Start")
    rnn = RecurrentPOSClass(10, [10, ])
    session = tf.InteractiveSession()
    rnn.set_session(session)
    vocab_size = len(word2idx)
    rnn.fit(vocab_size,
            X_train, Y_train, X_test, Y_test,
            recurr_unit=[GRUCell, ],
            nonlin_func=["relu", ],
            optimizer="adam",
            optimizer_args=(5*(1e-3), 0.9, 0.999),
            reg=1e-5,
            batch_sz=10,
            epochs=3)


