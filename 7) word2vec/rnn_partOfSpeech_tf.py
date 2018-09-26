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
from sklearn.metrics import f1_score
from Deep_NLP.reccurent_units import SimpleReccUnit, RateReccUnit, GateReccUnit, LSTM

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


    def model_initializer(self, V, K, recurr_unit, nonlin_func, optimizer, optimizer_args, reg):

        self.input_words = tf.placeholder(tf.int32, shape=(None, ), name="tfX")
        self.target_POS = tf.placeholder(tf.int32, shape=(None, ), name="tfT")
        self.StartPoints = tf.placeholder(tf.int32, shape=(None, ), name="stPoints")

        self.hidden_layers = []
        M_input = self.D

        self.W_embed = self.helper((V, self.D))

        for index, M_output in enumerate(self.hid_lay_sizes):
            self.hidden_layers.append(recurr_unit[index](M_input,
                                                         M_output,
                                                         nonlin_func[index]))
            M_input = M_output

        self.W_out = self.helper((M_input, K))

        self.params = [self.W_embed]
        for rec_unit in self.hidden_layers:
            self.params.append(rec_unit.get_params())
        self.params.append(self.W_out)

        Xw = tf.nn.embedding_lookup(self.W_embed[0], self.input_words)

        h_hidden = Xw
        for rec_unit in self.hidden_layers:
            h_hidden = rec_unit.output(h_hidden, self.StartPoints)

        logits = tf.matmul(h_hidden, self.W_out[0]) + self.W_out[1]
        self.prediction = tf.argmax(logits, axis=1)
        self.out_prob = tf.nn.softmax(logits)

        """  DO NOT APPLY REGULARIZATION TO BIAS TERMS """
        '''  self.params = [(W_embed, bias_embed), [...(W, b), ...first rec unit] , [...(W, b), ...second rec unit], ... (W_out, bias_out)] '''
        rcost = reg * sum(
            [tf.nn.l2_loss(coefs) for weight_and_bias in [self.params[0]] + [self.params[-1]] for coefs in
             weight_and_bias])
        rcost += reg * sum(
            [tf.nn.l2_loss(weights) for weight_and_bias in list(itertools.chain(*[i for i in self.params[1:-1]]))
             for weights in weight_and_bias[:1]])

        nce_weights = tf.transpose(self.W_out[0], [1, 0])  # needs to be VxD, not DxV
        nce_biases = self.W_out[1]
        h = tf.reshape(h_hidden, (-1, M_input))            # now M_input is size of last recurrent layer
        labels = tf.reshape(self.target_POS, (-1, 1))

        self.cost = tf.reduce_mean(
            tf.nn.sampled_softmax_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=labels,
                inputs=h,
                num_sampled=int(K/2),                            # number of negative samples
                num_classes=K)
                                  ) + rcost

        self.train_op = self.optimizer(optimizer, optimizer_args).minimize(self.cost)


    def fit(self, vocab_size, X_train, Y_train, X_test, Y_test, recurr_unit, nonlin_func, optimizer, optimizer_args, reg, batch_sz, epochs):

        ''' V - vocabulary size; K - output classes  '''
        V = vocab_size + 1
        K = len(set(itertools.chain(*Y_train)) | set(itertools.chain(*Y_test)))  # num classes

        self.model_initializer(V, K, recurr_unit, nonlin_func, optimizer, optimizer_args, reg)
        self.session.run(tf.global_variables_initializer())

        train_n_batches, test_n_batches = len(X_train) // batch_sz, len(X_test) // batch_sz
        costs = [0] * epochs
        train_accuracy, test_accuracy = [0] * epochs, [0] * epochs

        for epoch in range(epochs):
            t0 = datetime.datetime.now()
            X_train, Y_train = shuffle(X_train, Y_train)

            train_batch_acc = []
            for batch_n in range(train_n_batches):
                ''' sequenceLengths is needed to find where ends previous sentence and starts new one  '''
                X, Y = X_train[batch_n * batch_sz:(batch_n + 1) * batch_sz], Y_train[batch_n * batch_sz:(batch_n + 1) * batch_sz]
                sequenceLengths = list(itertools.chain(*[[1] + [0]*(n - 1) for n in list(map(len, X))]))
                X_flat, Y_flat = list(itertools.chain(*X)), list(itertools.chain(*Y))

                self.session.run(self.train_op, feed_dict={self.input_words: X_flat,
                                                           self.StartPoints: sequenceLengths,
                                                           self.target_POS: Y_flat})

                c, predict = self.session.run([self.cost, self.prediction], feed_dict={self.input_words: X_flat,
                                                                                       self.StartPoints: sequenceLengths,
                                                                                       self.target_POS: Y_flat})

                costs[epoch] += c

                ''' append tuple (num of correct predicted parts of speech, num of entire predictions) '''
                train_batch_acc.append((np.sum(np.array(Y_flat) == np.array(predict)), len(Y_flat)))


                if batch_n % 50 == 0:
                    sys.stdout.write("j/N: %d/%d/r" % (batch_n, train_n_batches))
                    sys.stdout.write('\n')
                    sys.stdout.flush()

            train_accuracy[epoch] = (lambda x: x[0] / x[1])(list(map(sum, zip(*train_batch_acc))))

            test_batch_acc = []
            for batch_n in range(test_n_batches):
                ''' sequenceLengths is needed to find where ends previous sentence and starts new one  '''
                X, Y = X_test[batch_n * batch_sz:(batch_n + 1) * batch_sz], Y_test[batch_n * batch_sz:(batch_n + 1) * batch_sz]
                sequenceLengths = list(itertools.chain(*[[1] + [0] * (n - 1) for n in list(map(len, X))]))
                X_flat, Y_flat = list(itertools.chain(*X)), list(itertools.chain(*Y))

                predict = self.session.run(self.prediction, feed_dict={self.input_words: X_flat,
                                                                       self.StartPoints: sequenceLengths,
                                                                       self.target_POS: Y_flat})

                ''' append tuple (num of correct predicted parts of speech, num of entire predictions) '''
                test_batch_acc.append((np.sum(np.array(Y_flat) == np.array(predict)), len(Y_flat)))

            test_accuracy[epoch] = (lambda x: x[0] / x[1])(list(map(sum, zip(*test_batch_acc))))

            print("epoch: ", epoch, \
                  " cost: ", costs[epoch], \
                  "train dataset correct rate: ", train_accuracy[epoch],
                  "test dataset correct rate: ", test_accuracy[epoch],
                  " time for epoch: ", (datetime.datetime.now() - t0))

        fig, ax1 = plt.subplots()
        ax1.plot(np.log(costs), 'b-')
        ax1.set_xlabel('epochs')
        # Make the y-axis label, ticks and tick labels match the line color.
        ax1.set_ylabel('log cost', color='b')

        ax2, ax3 = ax1.twinx(), ax1.twinx()
        ax2.plot(train_accuracy, 'r-')
        ax2.set_ylabel('train_accuracy', color='r')
        ax3.plot(test_accuracy, 'g-')


        fig.tight_layout()
        ax1.set_title('Log cost Curve' + "train_corr_rate: " + str(round(train_accuracy[-1], 3)) + \
                      "test_corr_rate: " + str(round(test_accuracy[-1], 3)) )
        ax1.grid()
        plt.show()




if __name__ == "__main__":)
    X_train, Y_train, X_test, Y_test, word2idx, tag2idx = get_pos_data(f_path('POS_train.txt'), f_path('POS_test.txt'))

    print("Start")
    rnn = RecurrentPOSClass(10, [10, ])
    session = tf.InteractiveSession()
    rnn.set_session(session)
    vocab_size = len(word2idx)
    rnn.fit(vocab_size,
            X_train, Y_train, X_test, Y_test,
            recurr_unit=[LSTM, ],
            nonlin_func=["relu", ],
            optimizer="adam",
            optimizer_args=(5*(1e-3), 0.9, 0.999),
            reg=1e-5,
            batch_sz=10,
            epochs=10)
