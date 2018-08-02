import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import nltk
import operator
import difflib
from sklearn.utils import shuffle
import sys
import datetime
import itertools
import warnings
import json
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import string
import random
import os

warnings.simplefilter(action='ignore', category=FutureWarning)
# to supress FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
# if saved_elem != "None":

f_name = lambda f_name: os.path.realpath(os.path.join(os.getcwd(), f_name)).replace('\\', '/')

KEEP_WORDS = set(['king', 'man', 'queen', 'woman',  'italy', 'rome', 'france', 'paris',  'london', 'britain', 'england',])

def get_sentence():
    ''' return list of lists containing sentences; each sentence - tokenized words and punktuation like ["i", "am", "," ...] '''
    return nltk.corpus.brown.sents()

def remove_punctuation(sentence):
    return sentence.translate(str.maketrans('', '', string.punctuation))

def index_sentence():
    word2idx = {'START': 0, 'END': 1}
    wordidx_count = {0: float('inf'), 1: float('inf')}
    sentences_idx = []
    for sentence in get_sentence():
        sentences_idx.append([])
        for word in [word for word in [remove_punctuation(s) for s in sentence] if word]: #remove punktuation and " " symbols::
            word = word.lower()
            if word not in word2idx:
                word2idx[word] = len(word2idx)
            wordidx_count[ word2idx[word] ] = wordidx_count.get( word2idx[word] , 0) + 1
            sentences_idx[-1].append(word2idx[word])
    print("Vocab size:", len(word2idx))
    ''' return list of sentences coded by unique numbers; dict word=>index; dict of word frequencies (words encoded with numbers) '''
    return sentences_idx, word2idx, wordidx_count

''' get indexed sentences with dictionary of words limited by N most frequently used words '''
def index_sentence_limit(vocab_size=2000, keep_words=KEEP_WORDS):
    sentences_idx, word2idx, wordidx_count = index_sentence()
    ind2word = {v: k for k, v in word2idx.items()}

    for word in keep_words:
        wordidx_count[ word2idx[word] ] = float('inf')
    sorted_wi_count = sorted(wordidx_count.items(), key=operator.itemgetter(1), reverse=True)

    word2idxSmall = {}
    old_new_idx = {}
    for idx, counter in sorted_wi_count[:vocab_size]:
        word2idxSmall[ ind2word[idx] ] = len(word2idxSmall)
        old_new_idx[idx] = word2idxSmall[ ind2word[idx] ]
    # let 'unknown' be the last token
    word2idxSmall['UNKNOWN'] = len(word2idxSmall)

    keep_words.update(set(['START', 'END']))
    for word in keep_words:
        assert (word in word2idxSmall)

    # map old idx to new idx
    small_sent_idx = []
    for sentence in sentences_idx:
        if len(sentence) > 1:
            small_sent_idx.append([old_new_idx.get(item, word2idxSmall['UNKNOWN'])  for item in sentence])

    return small_sent_idx, word2idxSmall

def init_weight_and_bias(M1, M2):
    return (np.random.randn(M1, M2) * np.sqrt(2.0 / M1)).astype(np.float32), (np.zeros(M2)).astype(np.float32)

    # return (np.random.randn(M1, M2) / np.sqrt(M2 + M1)).astype(np.float32), (np.zeros(M2)).astype(np.float32)
    # return (np.random.uniform(low=-np.sqrt(6/(M2 + M1)), high=np.sqrt(6/(M2 + M1)), size=(M1, M2))).astype(np.float32), \
    #         (np.zeros(M2)).astype(np.float32) #sigmoid
    # return (np.random.uniform(low=-4*np.sqrt(6/(M2 + M1)), high=4*np.sqrt(6/(M2 + M1)), size=(M1, M2))).astype(np.float32), \
    #         (np.zeros(M2)).astype(np.float32) #tanh


class SimpleReccUnit():
    def __init__(self, D, M2, nonlin_func, Wb_npz=None):
        self.D, self.M2 = D, M2
        self.nonlin_func = nonlin_func
        self.set_Wb(Wb_npz)

    def set_Wb(self, Wb_npz):
        if Wb_npz:
            self.Wx_h, self.Wh_h, self.Wx_0 = [list(map(tf.Variable, W_b)) for W_b in Wb_npz]
        else:
            self.Wx_h, self.Wh_h = [self.helper(dim) for dim in ((self.D, self.M2),
                                                                 (self.M2, self.M2))]
            #self.Wx_0 = self.helper((self.M2, self.M2)) #use only second bias term to initialize first "current_X" value of recurrence

        self.params = [self.Wx_h, self.Wh_h]         #, self.Wx_0]

    def get_params(self):
        return self.params

    def helper(self, dim):
        return list(map(tf.Variable, init_weight_and_bias(dim[0], dim[1])))

    def nonlinear(self, func):
        nonlin_dict = {"relu" : tf.nn.relu,
                       "tanh" : tf.nn.tanh,
                       "softmax" : tf.nn.softmax,
                       "sigmoid" : tf.nn.sigmoid,
                       "None" : lambda x: x}

        if func in nonlin_dict.keys():
            return nonlin_dict[func]


    def get_hidden_value(self, prev_h_rec, current_X):
        current_X = tf.reshape(current_X, (1, self.M2))
        prev_h_rec = tf.reshape(prev_h_rec, (1, self.M2))

        h_recur = self.nonlinear(self.nonlin_func)(
                                                   current_X + \
                                                   tf.matmul(tf.reshape(prev_h_rec, (1, self.M2)), self.Wh_h[0]) + \
                                                   self.Wh_h[1])

        return tf.reshape(h_recur, (self.M2,))


    def recurrence(self, prev_h_rec, current_X):
        result_h = tf.cond(
                           pred=tf.equal(current_X[1], tf.constant(1)),
                           true_fn=lambda: self.get_hidden_value(self.Wh_h[1], current_X[0]),
                           false_fn=lambda: self.get_hidden_value(prev_h_rec, current_X[0]))

                  #tf.cond(
                  #        pred=tf.equal(current_X[1], tf.constant(1)),
                  #        true_fn=lambda: self.get_hidden_value(self.Wx_0[1], current_X[0]),
                  #        false_fn=lambda: self.get_hidden_value(prev_h_rec, current_X[0]))

        return result_h

    def output(self, Xw, startPoints):
        x_h = tf.matmul(Xw, self.Wx_h[0])

        h_hidden = tf.scan(fn=self.recurrence,  # function to apply to each elems
                           elems=(x_h, startPoints),  # fn takes each of element in elems
                           initializer=self.Wh_h[1], )              #self.Wx_0[1], )

        return h_hidden



class RateReccUnit():
    def __init__(self, D, M2, nonlin_func, Wb_npz=None):
        self.D, self.M2 = D, M2
        self.nonlin_func = nonlin_func
        self.set_Wb(Wb_npz)

    def set_Wb(self, Wb_npz):
        if Wb_npz:
            self.Wx_h, self.Wx_z, self.Wh_z, self.Wh_h = [list(map(tf.Variable, W_b)) for W_b in Wb_npz]
        else:
            self.Wx_h, self.Wx_z, self.Wh_z, self.Wh_h = [self.helper(dim) for dim in ((self.D, self.M2),
                                                                                       (self.D, self.M2),
                                                                                       (self.M2, self.M2),
                                                                                       (self.M2, self.M2))]

        self.params = [self.Wx_h, self.Wx_z, self.Wh_z, self.Wh_h]

    def get_params(self):
        return self.params

    '''          =====================================       '''
    ''' {X} => || hHat_rec ->     z_g   ->  h_rec     ||    '''
    '''        ||      ^                    ^   |     ||     '''
    '''        ||      |        1 - z_g -> /    |     ||     '''
    '''        ||      |            ^           |     ||     '''
    '''        ||      |____________|___________|     ||     '''
    '''z_g = sigm(x.dot(Wx_z) + h_rec(t-1).dot(Whrec_zg) +bz)'''
    '''hHat_rec = f(x.dot(Wx_h_rec) + r * h_rec(t-1).dot(W_rec) + brec)'''
    '''h_rec = (1 - z_g) * h_rec(t-1) + z_g * hHat_rec'''

    def helper(self, dim):
        return list(map(tf.Variable, init_weight_and_bias(dim[0], dim[1])))

    def nonlinear(self, func):
        nonlin_dict = {"relu" : tf.nn.relu,
                       "tanh" : tf.nn.tanh,
                       "softmax" : tf.nn.softmax,
                       "sigmoid" : tf.nn.sigmoid,
                       "None" : lambda x: x}

        if func in nonlin_dict.keys():
            return nonlin_dict[func]

    def get_hidden_value(self, prev_h_rec, current_X):
        current_X = list(map(lambda x, y: tf.reshape(x, y), current_X, [(1, self.M2)] * 2))
        prev_h_rec = tf.reshape(prev_h_rec, (1, self.M2))

        hHat_recur = self.nonlinear(self.nonlin_func)(
                                                      current_X[0] + \
                                                      tf.matmul(prev_h_rec, self.Wh_h[0]) + \
                                                      self.Wh_h[1])

        z_gate = self.nonlinear("sigmoid")(
                                           current_X[1] + \
                                           tf.matmul(prev_h_rec, self.Wh_z[0]) + \
                                           self.Wh_z[1])

        h_recur = (1 - z_gate) * prev_h_rec + z_gate * hHat_recur
        return tf.reshape(h_recur, (self.M2,))

    def recurrence(self, prev_h_rec, current_X):
        result_h = tf.cond(
                           pred=tf.equal(current_X[1], tf.constant(1)),
                           true_fn=lambda: self.get_hidden_value(self.Wx_h[1], current_X[0]),
                           false_fn=lambda: self.get_hidden_value(prev_h_rec, current_X[0]))
        return result_h

    def output(self, Xw, startPoints):
        elems = list(map(lambda x, y: tf.matmul(x, y) [Xw]*2, [self.Wx_h[0], self.Wx_z[0]]))

        h_hidden = tf.scan(fn=self.recurrence,  # function to apply to each elems
                           elems=(elems, startPoints),  # fn takes each of element in elems
                           initializer=self.Wx_h[1], )
        return h_hidden


class GateReccUnit():
    def __init__(self, D, M2, nonlin_func, Wb_npz=None):
        self.D, self.M2 = D, M2
        self.nonlin_func = nonlin_func
        self.set_Wb(Wb_npz)

    def set_Wb(self, Wb_npz):
        if Wb_npz:
            self.Wx_h, self.Wx_z, self.Wh_z, self.Wh_h, self.Wx_r, self.Wh_r = [list(map(tf.Variable, W_b)) for W_b in Wb_npz]
        else:
            self.Wx_h, self.Wx_z, self.Wh_z, self.Wh_h, self.Wx_r, self.Wh_r = [self.helper(dim) for dim in ((self.D, self.M2),
                                                                                                             (self.D, self.M2),
                                                                                                             (self.M2, self.M2),
                                                                                                             (self.M2, self.M2),
                                                                                                             (self.D, self.M2),
                                                                                                             (self.M2, self.M2))]

        self.params = [self.Wx_h, self.Wx_z, self.Wh_z, self.Wh_h, self.Wx_r, self.Wh_r]


    def get_params(self):
        return self.params

    '''          =====================================    '''
    ''' {X} => || hHat_rec ->   z_g    ->  h_rec      ||  '''
    '''        ||      ^                    ^   |     ||  '''
    '''        ||      r        1 - z_g -> /    |     ||  '''
    '''        ||      ^             ^          |     ||  '''
    '''        ||      |_____________|__________|     ||  '''
    '''r = sigm(x.dot(Wx_r) + h_rec(t-1).dot(Whrec_r) +br) '''
    '''z_g = sigm(x.dot(Wx_z) + h_rec(t-1).dot(Whrec_zg) +bz)'''
    '''hHat_rec = f(x.dot(Wx_h_rec) + r * h_rec(t-1).dot(W_rec) + brec)'''
    '''h_rec = (1 - z_g) * h_rec(t-1) + z_g * hHat_rec'''

    def helper(self, dim):
        return list(map(tf.Variable, init_weight_and_bias(dim[0], dim[1])))

    def nonlinear(self, func):
        nonlin_dict = {"relu" : tf.nn.relu,
                       "tanh" : tf.nn.tanh,
                       "softmax" : tf.nn.softmax,
                       "sigmoid" : tf.nn.sigmoid,
                       "None" : lambda x: x}

        if func in nonlin_dict.keys():
            return nonlin_dict[func]

    def get_hidden_value(self, prev_h_rec, current_X):
        current_X = list(map(lambda x, y: tf.reshape(x, y), current_X, [(1, self.M2)] * 3))
        prev_h_rec = tf.reshape(prev_h_rec, (1, self.M2))

        r = self.nonlinear("sigmoid")(
                                      current_X[0] + \
                                      tf.matmul(prev_h_rec, self.Wh_r[0]) + \
                                      self.Wh_r[1])

        z_gate = self.nonlinear("sigmoid")(
                                           current_X[1] + \
                                           tf.matmul(prev_h_rec, self.Wh_z[0]) + \
                                           self.Wh_z[1])

        hHat_recur = self.nonlinear(self.nonlin_func)(
                                                      current_X[2] + \
                                                      r * tf.matmul(prev_h_rec, self.Wh_h[0]) + \
                                                      self.Wh_h[1])

        h_recur = (1 - z_gate) * prev_h_rec + z_gate * hHat_recur
        return tf.reshape(h_recur, (self.M2,))

    def recurrence(self, prev_h_rec, current_X):
        result_h = tf.cond(
                           pred=tf.equal(current_X[1], tf.constant(1)),
                           true_fn=lambda: self.get_hidden_value(self.Wx_h[1], current_X[0]),
                           false_fn=lambda: self.get_hidden_value(prev_h_rec, current_X[0]))
        return result_h

    def output(self, Xw, startPoints):
        elems = list(map(lambda x, y: tf.matmul(x, y), [Xw]*3, [self.Wx_r[0], self.Wx_z[0], self.Wx_h[0]]))

        h_hidden = tf.scan(fn=self.recurrence,  # function to apply to each elems
                           elems=(elems, startPoints),  # fn takes each of element in elems
                           initializer=self.Wx_h[1], )
        return h_hidden




class LSTM():
    def __init__(self, D, M2, nonlin_func, Wb_npz=None):
        self.D, self.M2 = D, M2
        self.nonlin_func = nonlin_func
        self.set_Wb(Wb_npz)

    def set_Wb(self, Wb_npz):
        if Wb_npz:
            self.Wx_i, self.Wc_i, self.Wh_i, self.Wx_f, self.Wc_f, self.Wh_f, self.Wx_c, self.Wh_c, self.Wx_o, self.Wc_o, self.Wh_o = \
                                                                                        [list(map(tf.Variable, W_b)) for W_b in Wb_npz]
        else:
            self.Wx_i, self.Wc_i, self.Wh_i = [self.helper(dim) for dim in ((self.D, self.M2),
                                                                            (self.M2, self.M2),
                                                                            (self.M2, self.M2))]
            self.Wx_f, self.Wc_f, self.Wh_f = [self.helper(dim) for dim in ((self.D, self.M2),
                                                                            (self.M2, self.M2),
                                                                            (self.M2, self.M2))]
            self.Wx_c, self.Wh_c = [self.helper(dim) for dim in ((self.D, self.M2),
                                                                 (self.M2, self.M2))]
            self.Wx_o, self.Wc_o, self.Wh_o = [self.helper(dim) for dim in ((self.D, self.M2),
                                                                            (self.M2, self.M2),
                                                                            (self.M2, self.M2))]

            #self.Wh_0, self.Wc_0 = [self.helper(dim) for dim in ((self.M2, self.M2),
            #                                                     (self.M2, self.M2))]



        self.params = [self.Wx_i, self.Wc_i, self.Wh_i, self.Wx_f, self.Wc_f, self.Wh_f, self.Wx_c, self.Wh_c, self.Wx_o, self.Wc_o, self.Wh_o]#, self.Wh_0, self.Wc_0]

    def get_params(self):
        return self.params

        '''          =====================================          '''
        '''   ____________________________________________          '''
        '''  |                       |_____        |     |          '''
        '''  |                       ^     |       ^     |          '''
        ''' {X} => ||    cHat   ->   i ->  c ----> o --> h      ||  '''
        '''        ||      ^               ^ |     ^     |      ||  '''
        '''        ||      |         f -> /  |     |_____|      ||  '''
        '''        ||      |         ^       |           |      ||  '''
        '''        ||      |_________|_______|___________|      ||  '''
        '''i = sigm(x.dot(Wx_i) + h_rec(t-1).dot(Whrec_i) + c(t-1).dot(Wc_i) + bi) '''
        '''f = sigm(x.dot(Wx_f) + h_rec(t-1).dot(Whrec_f) + c(t-1).dot(Wc_f) + bf) '''
        '''c = f*c(t-1) + i*tanh(x.dot(Wx_c) + h_rec(t-1).dot(Whrec_c) + bc)'''
        '''o = sigm(x.dot(Wx_o) + h_rec(t-1).dot(Whrec_o) + c.dot(Wc_o) + bo)'''
        '''h_rec = o*tanh(c)'''

    def helper(self, dim):
        return list(map(tf.Variable, init_weight_and_bias(dim[0], dim[1])))

    def nonlinear(self, func):
        nonlin_dict = {"relu" : tf.nn.relu,
                       "tanh" : tf.nn.tanh,
                       "softmax" : tf.nn.softmax,
                       "sigmoid" : tf.nn.sigmoid,
                       "None" : lambda x: x}

        if func in nonlin_dict.keys():
            return nonlin_dict[func]


    def get_hidden_value(self, prev_h_rec, current_X):
        current_X = list(map(lambda x, y: tf.reshape(x, y), current_X, [(1, self.M2)] * 4))

        prev_h_rec_ = tf.reshape(prev_h_rec[0], (1, self.M2))
        prev_c_rec_ = tf.reshape(prev_h_rec[1], (1, self.M2))

        inp_g = self.nonlinear("sigmoid")(
                                          current_X[0] + \
                                          tf.matmul(prev_h_rec_, self.Wh_i[0]) + \
                                          tf.matmul(prev_c_rec_, self.Wc_i[0]) + \
                                          self.Wc_i[1])

        forget_g = self.nonlinear("sigmoid")(
                                             current_X[1] + \
                                             tf.matmul(prev_h_rec_, self.Wh_f[0]) + \
                                             tf.matmul(prev_c_rec_, self.Wc_f[0]) + \
                                             self.Wc_f[1])

        c = forget_g * prev_c_rec_ + inp_g * self.nonlinear("tanh")(
                                                                    current_X[2] + \
                                                                    tf.matmul(prev_h_rec_, self.Wh_c[0]) + \
                                                                    self.Wh_c[1])

        out = self.nonlinear("sigmoid")(
                                        current_X[3] + \
                                        tf.matmul(prev_h_rec_, self.Wh_o[0]) + \
                                        tf.matmul(c, self.Wc_o[0]) + \
                                        self.Wc_o[1])

        h_recur = out * self.nonlinear("tanh")(c)

        ''' return tuple (h(t-1), c(t-1)) because recurrence in tf may have only 2 variables '''
        return (tf.reshape(h_recur, (self.M2,)), tf.reshape(c, (self.M2,)))

    def recurrence(self, prev_h_rec, current_X):
        result_h = tf.cond(
                           pred=tf.equal(current_X[1], tf.constant(1)),
                           true_fn=lambda: self.get_hidden_value((self.Wh_c[1], self.Wc_o[1]), current_X[0]),
                           false_fn=lambda: self.get_hidden_value(prev_h_rec, current_X[0]))
        return result_h

    def output(self, Xw, startPoints):
        elems = list(map(lambda x, y: tf.matmul(x, y), [Xw]*4, [self.Wx_i[0], self.Wx_f[0], self.Wx_c[0], self.Wx_o[0]]))

        h_hidden = tf.scan(fn=self.recurrence,          # function to apply to each elems
                           elems=(elems, startPoints),  # !!! CURRENT_X !!! fn takes each of element in elems = [Xw.dot(Wx_i[0]), Xw.dot(Wx_f[0]), Xw.dot(Wx_c[0]), Xw.dot(Wx_o[0])]
                           initializer=(self.Wh_c[1], self.Wc_o[1]), )
                           #initializer=(self.Wh_0[1], self.Wc_0[1]), )
        return h_hidden[0]






class RNN():
    '''  D - embedding size; V - vocabulary size '''
    def __init__(self, D, hid_lay_sizes, V):
        self.D, self.V = D, V
        self.hid_lay_sizes = hid_lay_sizes

    def nonlinear(self, func):
        nonlin_dict = {"relu" : tf.nn.relu,
                       "tanh" : tf.nn.tanh,
                       "softmax" : tf.nn.softmax,
                       "sigmoid" : tf.nn.sigmoid,
                       "None" : lambda x: x}

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

    def fun(self):
        return self.session.run(self.params[0][0])

    '''  recurr_unit, nonlin_func - lists'''
    def model_initializer(self, recurr_unit, nonlin_func, optimizer="adam", optimizer_args=(1e-5, 0.99, 0.999), reg=10e-3, lst_W_b=None):

        self.tfX = tf.placeholder(tf.int32, shape=(None,), name="tfX")
        self.tfT = tf.placeholder(tf.int32, shape=(None,), name="tfT")
        self.StartPoints = tf.placeholder(tf.int32, shape=(None,), name="stPoints")

        self.hidden_layers = []
        M_input = self.D
        if lst_W_b:
            unit_dict = {SimpleReccUnit : 2,
                         RateReccUnit : 4,
                         GateReccUnit : 6,
                         LSTM : 11}

            self.W_embed, self.W_out = lst_W_b[:2]
            counter = 2
            for index, unit in enumerate(recurr_unit):
                self.hidden_layers.append(unit(M_input,
                                               self.hid_lay_sizes[index],
                                               nonlin_func[index],
                                               Wb_npz=lst_W_b[counter: counter + unit_dict[unit]]))
                M_input = self.hid_lay_sizes[index]
                counter = counter + unit_dict[unit]
        else:
            self.W_embed = self.helper((self.V, self.D))
            for index, M_output in enumerate(self.hid_lay_sizes):
                self.hidden_layers.append(recurr_unit[index](M_input,
                                                             M_output,
                                                             nonlin_func[index]))
                M_input = M_output
            self.W_out = self.helper((M_input, self.V))

        self.params = [self.W_embed]
        for rec_unit in self.hidden_layers:
            self.params.append(rec_unit.get_params())
        self.params.append(self.W_out)

        Xw = tf.nn.embedding_lookup(self.W_embed[0], self.tfX)

        h_hidden = Xw
        for rec_unit in self.hidden_layers:
            h_hidden = rec_unit.output(h_hidden, self.StartPoints)

        logits = tf.matmul(h_hidden, self.W_out[0]) + self.W_out[1]
        self.prediction = tf.argmax(logits, axis=1)
        self.out_prob = tf.nn.softmax(logits)

        """  DO NOT APPLY REGULARIZATION TO BIAS TERMS """
        '''  self.params = [(W_embed, bias_embed), [...(W, b), ...first rec unit] , [...(W, b), ...second rec unit], ... (W_out, bias_out)] '''
        rcost = reg * sum([tf.nn.l2_loss(coefs) for weight_and_bias in [self.params[0]] + [self.params[-1]] for coefs in weight_and_bias])
        rcost += reg * sum([tf.nn.l2_loss(weights) for weight_and_bias in list(itertools.chain(*[i for i in self.params[1:-1]])) for weights in weight_and_bias[:1]])

        print(rcost, " REG: ", reg)


        nce_weights = tf.transpose(self.W_out[0], [1, 0])  # needs to be VxD, not DxV
        nce_biases = self.W_out[1]
        h = tf.reshape(h_hidden, (-1, M_input)) # now M_input is size of last recurrent layer
        labels = tf.reshape(self.tfT, (-1, 1))

        ''' This is a faster way to train a softmax classifier over a huge number of classes '''
        self.cost = tf.reduce_mean(
            tf.nn.sampled_softmax_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=labels,
                inputs=h,
                num_sampled=50,  # number of negative samples
                num_classes=self.V
            )
        ) + rcost

        self.train_op = self.optimizer(optimizer, optimizer_args).minimize(self.cost)

    """
    def fit(self, X, Y, sqcLengths, recurr_unit, nonlin_func, optimizer, optimizer_args, reg, batch_sz=500, epochs=500, show_fig=False):

        self.model_initializer(recurr_unit, nonlin_func, optimizer, optimizer_args, reg)

        self.session.run(tf.global_variables_initializer())

        n_batches = np.where(sqcLengths == 1)[0].shape[0] // batch_sz
        costs = [0] * epochs
        final_correct_rate = 0
        for epoch in range(epochs):
            t0 = datetime.datetime.now()
            ''' for each sentence we learn to guess next word to each current word '''
            corr_rate = []
            start_sent_ind = np.where(sqcLengths == 1)[0]

            for batch_num in range(n_batches):
                # sequenceLengths, inp_data_sequence, target_data_sequence = np.array([]), [], []
                sequenceLengths, inp_data_sequence, target_data_sequence = np.array([]), np.array([]), np.array([])

                inp_data_sequence = X[start_sent_ind[batch_num * batch_sz] : start_sent_ind[(batch_num + 1) * batch_sz]]
                target_data_sequence = Y[start_sent_ind[batch_num * batch_sz] : start_sent_ind[(batch_num + 1) * batch_sz]]
                sequenceLengths = sqcLengths[start_sent_ind[batch_num * batch_sz] : start_sent_ind[(batch_num + 1) * batch_sz]]

                self.session.run(self.train_op, feed_dict={self.tfX: inp_data_sequence, self.StartPoints: sequenceLengths, self.tfT: target_data_sequence})
                c, predict = self.session.run([self.cost, self.prediction], 
                                              feed_dict={self.tfX: inp_data_sequence, self.StartPoints: sequenceLengths, self.tfT: target_data_sequence})

                costs[epoch] += c

                ''' calculate similarity of two sentences - real and predicted '''
                corr_rate.append(difflib.SequenceMatcher(None, predict, target_data_sequence).ratio())
                if batch_num % 10 == 0:
                    sys.stdout.write("j/N: %d/%d/r" % (batch_num, n_batches))
                    sys.stdout.write('\n')
                    sys.stdout.flush()

            print("epoch: ", epoch, " cost: ", costs[epoch], " correct rate: ", sum(corr_rate) / len(corr_rate),
                  " time for epoch: ", (datetime.datetime.now() - t0))

            if epoch == epochs - 1:
                final_correct_rate = sum(corr_rate) / len(corr_rate)

            if sum(corr_rate) / len(corr_rate) >= 0.91 or corr_rate[-1] == None:
                final_correct_rate = sum(corr_rate[:-1]) / len(corr_rate[:-1])
                break

        fig = plt.figure()
        ax = plt.subplot(111)
        ax.plot(np.log(costs))
        ax.set_title('Log cost Curve' + " corr_rate: " + str(final_correct_rate))
        ax.grid()
        if show_fig:
            plt.show()
        else:
            name_dict = {LSTM: "LSTM", 
                         GateReccUnit: "GateReccUnit", 
                         RateReccUnit: "RateReccUnit", 
                         SimpleReccUnit: "SimpleReccUnit"}
                         
            name = "Brown_" + \ 
                   name_dict[recurr_unit[0]] + \
                   optimizer + \
                   "_".join(map(str, optimizer_args)) + \
                   str(epochs) + \
                   "_reg_" + \
                   str(reg) + \
                   ".png"
                   
            fig.savefig(name)
    """

    def fit_(self, X, recurr_unit, nonlin_func, optimizer, optimizer_args, reg, batch_sz = 200, epochs=500, show_fig=False):

        self.model_initializer(recurr_unit, nonlin_func, optimizer, optimizer_args, reg)

        self.session.run(tf.global_variables_initializer())

        n_batches = len(X) // batch_sz
        costs = [0] * epochs
        accuracy = [0] * epochs

        for epoch in range(epochs):
            t0 = datetime.datetime.now()
            X = shuffle(X)
            ''' for each sentence we learn to guess next word to each current word '''

            corr_words, total_words = 0, 0
            acc_lst = []
            for batch_num in range(n_batches):
                #sequenceLengths, inp_data_sequence, target_data_sequence = np.array([]), [], []
                sequenceLengths, inp_data_sequence, target_data_sequence = np.array([]), np.array([]), np.array([])
                for k in range(batch_num * batch_sz, (batch_num + 1) * batch_sz):
                    if np.random.random() < 0.1:
                        #inp_data_sequence.extend([0] + X[k])
                        #target_data_sequence.extend(X[k] + [1])
                        inp_data_sequence = np.append(inp_data_sequence, [0] + X[k])
                        target_data_sequence = np.append(target_data_sequence, X[k] + [1])
                        sequenceLengths = np.append(sequenceLengths, [1] + [0]*len(X[k]))
                    else:
                        #inp_data_sequence.extend([0] + X[k][:-1])
                        #target_data_sequence.extend(X[k])
                        inp_data_sequence = np.append(inp_data_sequence, [0] + X[k][:-1])
                        target_data_sequence = np.append(target_data_sequence, X[k])
                        sequenceLengths = np.append(sequenceLengths, [1] + [0]*len(X[k][:-1]))

                self.session.run(self.train_op, feed_dict={self.tfX: inp_data_sequence,
                                                           self.StartPoints: sequenceLengths,
                                                           self.tfT: target_data_sequence})
                c, predict = self.session.run([self.cost, self.prediction], feed_dict={self.tfX: inp_data_sequence,
                                                                                       self.StartPoints: sequenceLengths,
                                                                                       self.tfT: target_data_sequence})

                costs[epoch] += c
                corr_words += np.sum(np.array(target_data_sequence) == np.array(predict))
                total_words += target_data_sequence.shape[0]
                acc_lst.append(np.sum(np.array(target_data_sequence) == np.array(predict)) * 1.0 / target_data_sequence.shape[0])

                #if batch_num % 5 == 0:
                #    print("target: ", type(target_data_sequence[1]), target_data_sequence[:1][0])
                #    print("predict: ", type(predict[1]), predict[:1][0])
                if batch_num % 50 == 0:
                    sys.stdout.write("j/N: %d/%d/r" % (batch_num,  n_batches))
                    sys.stdout.write('\n')
                    sys.stdout.flush()

            ''' calculate similarity of two sentences - real and predicted '''
            accuracy[epoch] = corr_words * 1.0 / total_words
            print("epoch: ", epoch, " cost: ", costs[epoch], " correct rate: ", accuracy[epoch],
                  sum(acc_lst) / len(acc_lst), " time for epoch: ", (datetime.datetime.now() - t0))

            if accuracy[epoch] >= 0.91 or accuracy[epoch] == None:
                break

        fig, ax1 = plt.subplots()
        ax1.plot(np.log(costs), 'b-')
        ax1.set_xlabel('epochs')
        # Make the y-axis label, ticks and tick labels match the line color.
        ax1.set_ylabel('log cost', color='b')

        ax2 = ax1.twinx()
        ax2.plot(accuracy, 'r-')
        ax2.set_ylabel('accuracy', color='r')

        fig.tight_layout()
        ax1.set_title('Log cost Curve' + " corr_rate: " + str(accuracy[-1]))
        ax1.grid()
        if show_fig:
            plt.show()
        else:
            name_dict = {LSTM: "LSTM",
                         GateReccUnit: "GateReccUnit",
                         RateReccUnit: "RateReccUnit",
                         SimpleReccUnit: "SimpleReccUnit"}

            name = "BatchBrown_" + \
                   str(self.D) + \
                   "_".join(map(str, self.hid_lay_sizes)) + \
                   "_" + name_dict[recurr_unit[0]] + \
                   nonlin_func[0] + \
                   optimizer + \
                   "_".join(map(str, optimizer_args)) + \
                   str(epochs) + \
                   "_reg_" + \
                   str(reg) + \
                   ".png"

            fig.savefig(name)


    def save(self, filename):
        actual_params = self.session.run(self.params)
        saved_lst = [p for Wb in actual_params[::len(actual_params)-1] for p in Wb] # some_list[::len(some_list)-1] - get first and last element
        for hid_layer in actual_params[1:-1]:
            saved_lst.extend([p for Wb in hid_layer for p in Wb])
            saved_lst.append("None")
        '''  list with params: [W_embed, b_embed, W_out, b_out, ...W_b_layer_1..., "None", ...W_b_layer_2..., "None", ...] 
             - first 4 W and b in and out params than parameters of each layer separated by "None" - to know the start position of each new layer'''
        np.savez(filename, *saved_lst)

    @staticmethod
    def load(filename, activation):
        ''' TODO: would prefer to save activation to file too '''
        npz = np.load(f_name(filename))
        param_num = len([elem for elem in npz])
        mylist = [npz['arr_' + str(n)] for n in range(param_num)]
        '''  create lists of 1)hid_layer_sizes 2)lst_W_b - saved matricies list of tuples 3)list of recurrent units  '''
        lst_W_b = list(zip(mylist[0:4:2], mylist[1:4:2])) # [(W_embed, b_embed), (W_out, b_out)]
        hid_lay_sizes = []
        recurr_unit = []

        V, D = npz['arr_0'].shape

        unit = {4: SimpleReccUnit, 8: RateReccUnit, 12: GateReccUnit, 22: LSTM}
        buffer = []
        for saved_elem in mylist[4:]:
            if saved_elem != "None":
                buffer.append(saved_elem)
            else:
                recurr_unit.append(unit[len(buffer)])
                lst_W_b.extend(list(zip(buffer[0::2], buffer[1::2])))
                hid_lay_sizes.append(buffer[0].shape[1]) # add M2 size of each new hidden layer
                buffer = []
        #print("V: ", V, "D: ", D)
        #print("hid_lay_sizes: ", hid_lay_sizes)
        #print("recurr_unit: ", recurr_unit)
        #for i in lst_W_b:
        #    for j in i:
        #        print(j.shape)
        rnn = RNN(D=D, hid_lay_sizes=hid_lay_sizes, V=V)
        rnn.model_initializer(recurr_unit, nonlin_func=activation, lst_W_b=lst_W_b)

        return rnn


    def predict(self, prev_words, startPoints):
        # don't use argmax, so that we can sample from this probability distribution
        self.session.run(tf.global_variables_initializer())
        return self.session.run(self.out_prob, feed_dict={self.tfX: prev_words, self.StartPoints: startPoints})


    def generate(self, word2index):
        index_to_word = {index:word for word, index in word2index.items()}
        words_id = len(index_to_word)
        words, startPoints = [0], [1]
        num_line = 0
        ''' it will be 10 lines '''
        while num_line < 10:
            choose_word_idx = np.random.choice(words_id, p=self.predict(words, startPoints)[-1])
            words.append(choose_word_idx)
            ''' if  choose_word_idx != 0 or 1 - not start or end token'''
            if choose_word_idx > 1:
                startPoints.append(0)
                print(index_to_word[choose_word_idx], end=" ")
                ''' end token '''
            elif choose_word_idx == 1:
                num_line += 1
                words, startPoints = [0], [1]
                print("END LINE")
                print("")


def find_analogies(w1, w2, w3, word2ind_file, W_b_file):
    W_embed = np.load(f_name(W_b_file))['arr_0']
    with open(word2ind_file) as f:
        word2idx = json.load(f_name(f))

    word_1 = W_embed[word2idx[w1]]
    word_2 = W_embed[word2idx[w2]]
    word_3 = W_embed[word2idx[w3]]
    v0 = word_1 - word_2 + word_3

    def dist1(a, b):
        return np.linalg.norm(a - b)

    def dist2(a, b):
        return 1 - a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))

    for dist, name in [(dist1, 'Euclidean'), (dist2, 'cosine')]:
        min_dist = float('inf')
        best_word = ''
        for word, idx in word2idx.items():
            if word not in (w1, w2, w3):
                v1 = W_embed[idx]
                d = dist(v0, v1)
                if d < min_dist:
                    min_dist = d
                    best_word = word
        print("closest match by", name, "distance:", best_word)
        print(w1, "-", w2, "=", best_word, "-", w3)

''' model = PCA, TSNE '''
def visualize(word2ind_file, W_b_file, model):
    W_embed = np.load(f_name(W_b_file))['arr_0']
    with open(word2ind_file) as f:
        word2idx = json.load(f_name(f))

    V, D = W_embed.shape
    model = model()
    Z = model.fit_transform(W_embed)

    #get index(word) dict
    id_word = {v : k for k, v in word2idx.items()}

    fig, ax = plt.subplots()
    ax.scatter(Z[:, 0], Z[:, 1])

    for i in range(V):
        ax.annotate(id_word[i], (Z[i, 0], Z[i, 1]))
    plt.show()


if __name__ == "__main__":
    small_sent_idx, word2idxSmall = index_sentence_limit(vocab_size=40000)
    print(len(word2idxSmall))
    #sent_idx = list(filter(lambda x: x.count(3000) < len(x) * 1.0 / 20, small_sent_idx))  # keep only sentences with not too much "UNKNOWN" words
    print("num of sentences: ", len(small_sent_idx))
    random.seed(123)
    selected_num = random.sample(range(len(small_sent_idx)), 5000)
    search_dataset = list(np.array(small_sent_idx)[selected_num])
    print("search_dataset num of sentences: ", len(search_dataset))
    print(search_dataset[0])
    """  for search_dataset embed 100 LSTM 250 relu adam (5*(1e-3), 0.9, 0.999) reg=1e-7 epochs=60 - 100 shows up to 45% acc """

    """ this part for fit() method """
    '''
    X, Y, sequenceLengths = np.array([]), np.array([]), np.array([])
    sent_idx = shuffle(sent_idx)
    for sentence in sent_idx:
        if np.random.random() < 0.1:
            X = np.append(X, [0] + sentence)
            Y = np.append(Y, sentence + [1])
            sequenceLengths = np.append(sequenceLengths, [1] + [0] * len(sentence))
        else:
            X = np.append(X, [0] + sentence[:-1])
            Y = np.append(Y, sentence)
            sequenceLengths = np.append(sequenceLengths, [1] + [0] * len(sentence[:-1]))
    '''


    print("Start")
    rnn = RNN(150, [250, ], len(word2idxSmall))
    session = tf.InteractiveSession()
    rnn.set_session(session)
    rnn.fit_(X=small_sent_idx, recurr_unit=[LSTM, ],
             nonlin_func=["relu", ], optimizer="adam",
             optimizer_args=(5*(1e-3), 0.9, 0.999), reg=1e-5,
             epochs=100, show_fig=False)
    #rnn.fit(X=X, Y=Y, sqcLengths=sequenceLengths,
    #        recurr_unit=[SimpleReccUnit, ], nonlin_func=["relu", ],
    #        optimizer="adam", optimizer_args=(1e-4, 0.99, 0.999),
    #        reg=1e-10, epochs=100, show_fig=False)


    rnn.save("Batch200_embed_150LSTM_adam_relu_250_vocab_40000_FULL_5mult1e3_reg1e5epoch_100.npz")
    with open("brown_word2idx_40000.json", 'w') as f:
        json.dump(word2idxSmall, f)
    #rnn.generate(word2idxSmall)

    loaded_recurrent_model = RNN.load(filename="Batch_embed_100LSTM_adam_relu_350_vocab_3000_10_part.npz", activation=["relu", ])
    session = tf.InteractiveSession()
    loaded_recurrent_model.set_session(session)
    #loaded_recurrent_model.generate(word2idxSmall)
    W_embed = rnn.fun()

    V, D = rnn.fun().shape
    model_1 = PCA()
    model_2 = TSNE()
    Z_1 = model_1.fit_transform(W_embed)
    Z_2 = model_2.fit_transform(W_embed)

    # get index(word) dict
    id_word = {v: k for k, v in word2idxSmall.items()}

    fig, ax = plt.subplots()
    ax.scatter(Z_1[:, 0], Z_1[:, 1])

    for i in range(V):
        ax.annotate(id_word[i], (Z_1[i, 0], Z_1[i, 1]))
    plt.show()

    fig, ax = plt.subplots()
    ax.scatter(Z_2[:, 0], Z_2[:, 1])

    for i in range(V):
        ax.annotate(id_word[i], (Z_2[i, 0], Z_2[i, 1]))
    plt.show()

    for words in [('king', 'man', 'woman'), ('france', 'paris', 'london'), ('france', 'paris', 'rome'), ('paris', 'france', 'italy')]:
        word_1 = W_embed[word2idxSmall[words[0]]]
        word_2 = W_embed[word2idxSmall[words[1]]]
        word_3 = W_embed[word2idxSmall[words[2]]]
        v0 = word_1 - word_2 + word_3


        def dist1(a, b):
            return np.linalg.norm(a - b)


        def dist2(a, b):
            return 1 - a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))


        for dist, name in [(dist1, 'Euclidean'), (dist2, 'cosine')]:
            min_dist = float('inf')
            best_word = ''
            for word, idx in word2idxSmall.items():
                if word not in words:
                    v1 = W_embed[idx]
                    d = dist(v0, v1)
                    if d < min_dist:
                        min_dist = d
                        best_word = word
            print("closest match by", name, "distance:", best_word)
            print(words[0], "-", words[1], "=", best_word, "-", words[2])
   

    for words in [('king', 'man', 'woman'), ('france', 'paris', 'london'), ('france', 'paris', 'rome'), ('paris', 'france', 'italy')]:
        find_analogies(*words, word2ind_file="brown_word2idx_full.json", W_b_file="Batch200_embed_150LSTM_adam_relu_250_vocab_40000_FULL_5mult1e3_reg1e5epoch_100.npz")

    visualize(word2ind_file="brown_word2idx_40000.json", W_b_file="Batch200_embed_150LSTM_adam_relu_250_vocab_40000_FULL_5mult1e3_reg1e5epoch_100.npz", model=PCA)
    visualize(word2ind_file="brown_word2idx_40000.json", W_b_file="Batch200_embed_150LSTM_adam_relu_250_vocab_40000_FULL_5mult1e3_reg1e5epoch_100.npz", model=TSNE)



