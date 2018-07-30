import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import string
from sklearn.utils import shuffle
import difflib
import pickle

f = lambda f_name: os.path.realpath(os.path.join(os.getcwd(), f_name)).replace('\\', '/')

def init_weight_and_bias(M1, M2):
    #return (np.random.randn(M1, M2) * np.sqrt(2.0 / M1)).astype(np.float32), (np.zeros(M2)).astype(np.float32)
    return (np.random.randn(M1, M2) / np.sqrt(M2 + M1)).astype(np.float32), (np.zeros(M2)).astype(np.float32)

def remove_punctuation(sentence):
    return sentence.translate(str.maketrans('', '', string.punctuation))

def get_poetry_data():
    word2idx = {'START': 0, 'END': 1}
    current_idx = 2
    sentences = []
    for line in open(f('edgar_allan_poe.txt'), encoding="utf8"):
        line = line.strip()
        if line:
            tokens = remove_punctuation(line.lower()).split()
            sentence = []
            for t in tokens:
                if t not in word2idx:
                    word2idx[t] = current_idx
                    current_idx += 1
                idx = word2idx[t]
                sentence.append(idx)
            sentences.append(sentence)
    ''' SO WE RETURN LIST OF WORD SEQUENCES AND ENCODED DICTIONARY WORD => UNIQUE NUMBER '''
    return sentences, word2idx



class SimpleReccUnit():
    def __init__(self, D, M2, nonlin_func, Wb_npz=None):
        self.D, self.M2 = D, M2
        self.nonlin_func = nonlin_func
        self.set_Wb(Wb_npz)

    def set_Wb(self, Wb_npz):
        if Wb_npz:
            self.Wx_h, self.Wh_h = [list(map(tf.Variable, W_b)) for W_b in Wb_npz]
        else:
            self.Wx_h, self.Wh_h = [self.helper(dim) for dim in ((self.D, self.M2), (self.M2, self.M2))]

        self.params = [self.Wx_h, self.Wh_h]

    def get_params(self):
        return self.params

    def helper(self, dim):
        return list(map(tf.Variable, init_weight_and_bias(dim[0], dim[1])))

    def nonlinear(self, func):
        nonlin_dict = {"relu" : tf.nn.relu, "tanh" : tf.nn.tanh, "softmax" : tf.nn.softmax, "sigmoid" : tf.nn.sigmoid, "None" : lambda x: x}
        if func in nonlin_dict.keys():
            return nonlin_dict[func]

    def recurrence(self, prev_h_rec, current_X):
        current_X = tf.reshape(current_X, (1, self.D))
        prev_h_rec = tf.reshape(prev_h_rec, (1, self.M2))

        h_recur = self.nonlinear(self.nonlin_func)(  tf.matmul(current_X, self.Wx_h[0])  + \
                                                     tf.matmul(tf.reshape(prev_h_rec, (1, self.M2)), self.Wh_h[0]) +  self.Wh_h[1])
        return tf.reshape(h_recur, (self.M2,))

    def output(self, Xw):
        h_hidden = tf.scan(fn=self.recurrence,  # function to apply to each elems
                           elems=Xw,  # fn takes each of element in elems
                           initializer=self.Wx_h[1], )
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
            self.Wx_h, self.Wx_z, self.Wh_z, self.Wh_h = [self.helper(dim) for dim in ((self.D, self.M2), (self.D, self.M2), 
                                                                                       (self.M2, self.M2), (self.M2, self.M2))]

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
        nonlin_dict = {"relu" : tf.nn.relu, "tanh" : tf.nn.tanh, "softmax" : tf.nn.softmax, "sigmoid" : tf.nn.sigmoid, "None" : lambda x: x}
        if func in nonlin_dict.keys():
            return nonlin_dict[func]

    def recurrence(self, prev_h_rec, current_X):
        current_X = tf.reshape(current_X, (1, self.D))
        prev_h_rec = tf.reshape(prev_h_rec, (1, self.M2))

        hHat_recur = self.nonlinear(self.nonlin_func)(tf.matmul(current_X, self.Wx_h[0]) + tf.matmul(prev_h_rec, self.Wh_h[0]) + self.Wh_h[1])
        z_gate = self.nonlinear("sigmoid")(tf.matmul(current_X, self.Wx_z[0]) + tf.matmul(prev_h_rec, self.Wh_z[0]) + self.Wh_z[1])
        h_recur = (1 - z_gate) * prev_h_rec + z_gate * hHat_recur

        return tf.reshape(h_recur, (self.M2,))

    def output(self, Xw):
        h_hidden = tf.scan(fn=self.recurrence,  # function to apply to each elems
                           elems=Xw,  # fn takes each of element in elems
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
            self.Wx_h, self.Wx_z, self.Wh_z, self.Wh_h, self.Wx_r, self.Wh_r = [self.helper(dim) for dim in
                                                                                ((self.D, self.M2), (self.D, self.M2), (self.M2, self.M2), 
                                                                                 (self.M2, self.M2), (self.D, self.M2), (self.M2, self.M2))]

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
        nonlin_dict = {"relu" : tf.nn.relu, "tanh" : tf.nn.tanh, "softmax" : tf.nn.softmax, "sigmoid" : tf.nn.sigmoid, "None" : lambda x: x}
        if func in nonlin_dict.keys():
            return nonlin_dict[func]

    def recurrence(self, prev_h_rec, current_X):
        current_X = tf.reshape(current_X, (1, self.D))
        prev_h_rec = tf.reshape(prev_h_rec, (1, self.M2))

        r = self.nonlinear("sigmoid")(tf.matmul(current_X, self.Wx_r[0]) + tf.matmul(prev_h_rec, self.Wh_r[0]) + self.Wh_r[1])
        z_gate = self.nonlinear("sigmoid")(tf.matmul(current_X, self.Wx_z[0]) + tf.matmul(prev_h_rec, self.Wh_z[0]) + self.Wh_z[1])
        hHat_recur = self.nonlinear(self.nonlin_func)(tf.matmul(current_X, self.Wx_h[0]) + r * tf.matmul(prev_h_rec, self.Wh_h[0]) + self.Wh_h[1])
        h_recur = (1 - z_gate) * prev_h_rec + z_gate * hHat_recur

        return tf.reshape(h_recur, (self.M2,))

    def output(self, Xw):
        h_hidden = tf.scan(fn=self.recurrence,  # function to apply to each elems
                           elems=Xw,  # fn takes each of element in elems
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
            self.Wx_i, self.Wc_i, self.Wh_i = [self.helper(dim) for dim in ((self.D, self.M2), (self.M2, self.M2), (self.M2, self.M2))]
            self.Wx_f, self.Wc_f, self.Wh_f = [self.helper(dim) for dim in ((self.D, self.M2), (self.M2, self.M2), (self.M2, self.M2))]
            self.Wx_c, self.Wh_c = [self.helper(dim) for dim in ((self.D, self.M2), (self.M2, self.M2))]
            self.Wx_o, self.Wc_o, self.Wh_o = [self.helper(dim) for dim in ((self.D, self.M2), (self.M2, self.M2), (self.M2, self.M2))]

        self.params = [self.Wx_i, self.Wc_i, self.Wh_i, self.Wx_f, self.Wc_f, self.Wh_f, self.Wx_c, self.Wh_c, self.Wx_o, self.Wc_o, self.Wh_o]

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
        nonlin_dict = {"relu" : tf.nn.relu, "tanh" : tf.nn.tanh, "softmax" : tf.nn.softmax, "sigmoid" : tf.nn.sigmoid, "None" : lambda x: x}
        if func in nonlin_dict.keys():
            return nonlin_dict[func]

    def recurrence(self, prev_h_c, current_X):
        current_X = tf.reshape(current_X, (1, self.D))
        prev_h_rec = tf.reshape(prev_h_c[0], (1, self.M2))
        prev_c_rec = tf.reshape(prev_h_c[1], (1, self.M2))

        inp_g = self.nonlinear("sigmoid")(tf.matmul(current_X, self.Wx_i[0]) + tf.matmul(prev_h_rec, self.Wh_i[0]) + \
                                          tf.matmul(prev_c_rec, self.Wc_i[0]) + self.Wc_i[1])
        forget_g = self.nonlinear("sigmoid")(tf.matmul(current_X, self.Wx_f[0]) + tf.matmul(prev_h_rec, self.Wh_f[0]) + \ 
                                             tf.matmul(prev_c_rec, self.Wc_f[0]) + self.Wc_f[1])
        c = forget_g * prev_c_rec + inp_g * self.nonlinear("tanh")(tf.matmul(current_X, self.Wx_c[0]) + \
                                                                   tf.matmul(prev_h_rec, self.Wh_c[0]) + self.Wh_c[1])
        out = self.nonlinear("sigmoid")(tf.matmul(current_X, self.Wx_o[0]) + tf.matmul(prev_h_rec, self.Wh_o[0]) + \ 
                                                                             tf.matmul(c, self.Wc_o[0]) + self.Wc_o[1])
        h_recur = out * self.nonlinear("tanh")(c)

        ''' return tuple (h(t-1), c(t-1)) because recurrence in tf may have only 2 variables '''
        return (tf.reshape(h_recur, (self.M2,)), tf.reshape(c, (self.M2,)))

    def output(self, Xw):
        h_hidden = tf.scan(fn=self.recurrence,  # function to apply to each elems
                           elems=Xw,  # fn takes each of element in elems
                           initializer=(self.Wh_c[1], self.Wc_o[1]), )
        return h_hidden[0]




class GeneratePoetry():
    '''  D - dimension of vector of word embedding matrix, M2 - num of recurrent hidden units layer, V - size of vocabulary: 
             simply - all words from sample text'''
    def __init__(self, D, M2, V):
        self.D, self.M2, self.V = D, M2, V

    def nonlinear(self, func):
        nonlin_dict = {"relu" : tf.nn.relu, "tanh" : tf.nn.tanh, "softmax" : tf.nn.softmax, "sigmoid" : tf.nn.sigmoid, "None" : lambda x: x}
        if func in nonlin_dict.keys():
            return nonlin_dict[func]

    def set_session(self, session):
        self.session = session

    def optimizer(self, auto_optimizer, auto_opt_args):
        optimizer_dict = {"graddes": tf.train.GradientDescentOptimizer, "adadelta": tf.train.AdadeltaOptimizer, "adagrad": tf.train.AdagradOptimizer,
                          "adagradD": tf.train.AdagradDAOptimizer, "momentum": tf.train.MomentumOptimizer, "adam": tf.train.AdamOptimizer,
                          "ftlr": tf.train.FtrlOptimizer, "proxgrad": tf.train.ProximalGradientDescentOptimizer, 
                          "proxadagrad": tf.train.ProximalAdagradOptimizer, "rms": tf.train.RMSPropOptimizer}

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

    def model_initializer(self, recurr_unit, nonlin_func, optimizer="adam", optimizer_args=(1e-5, 0.99, 0.999), reg=10-3, train_mode=True, lst_W_b=None):

        self.tfX = tf.placeholder(tf.int32, shape=(None,), name="tfX")
        self.tfT = tf.placeholder(tf.int32, shape=(None,), name="tfT")

        if train_mode:
            self.W_embed, self.W_out = [self.helper(dim) for dim in ((self.V, self.D), (self.M2, self.V))]
            self.RecUnit = recurr_unit(self.D, self.M2, nonlin_func)
            #self.params = [self.W_embed, self.W_out] + self.RecUnit.get_params()
        else:
            self.W_embed, self.W_out = lst_W_b[ :2]
            self.RecUnit = recurr_unit(self.D, self.M2, nonlin_func, Wb_npz=lst_W_b[2: ])
        self.params = [self.W_embed, self.W_out] + self.RecUnit.get_params()

        # convert word indexes to word vectors - this would be equivalent to doing - We[tfX] in Numpy - or: X_one_hot = one_hot_encode(X) X_one_hot.dot(We)
        ''' embedding_lookup function retrieves rows of the params tensor. The behavior is similar to using indexing with arrays in numpy. E.g.
            matrix = np.random.random([1024, 64])  # 64-dimensional embeddings => ids = np.array([0, 5, 17, 33]) => print matrix[ids]  
            # prints a matrix of shape [4, 64] '''
        Xw = tf.nn.embedding_lookup(self.W_embed[0], self.tfX)


        h_hidden = self.RecUnit.output(Xw)
        # get output: logits has shape (num of words in sentence X size of vocabulary); self.prediction has shape (num of words in sentence, ); 
        # self.out_prob has shape (num of words in sentence X size of vocabulary)
        logits = tf.matmul(h_hidden, self.W_out[0]) + self.W_out[1]
        self.prediction = tf.argmax(logits, axis=1)
        self.out_prob = tf.nn.softmax(logits)

        rcost = reg * sum([tf.nn.l2_loss(coefs) for weight_and_bias in self.params for coefs in weight_and_bias])
        # cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.tfT)) + rcost

        nce_weights = tf.transpose(self.W_out[0], [1, 0])  # needs to be VxD, not DxV
        nce_biases = self.W_out[1]
        h = tf.reshape(h_hidden, (-1, self.M2))
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

    def fit(self, X, recurr_unit, nonlin_func, optimizer, optimizer_args, reg, epochs=500, show_fig=False):

        self.model_initializer(recurr_unit, nonlin_func, optimizer, optimizer_args, reg)

        self.session.run(tf.global_variables_initializer())

        costs = [0]*epochs
        final_correct_rate = 0
        for epoch in range(epochs):
            X = shuffle(X)
            ''' for each sentence we learn to guess next word to each current word '''
            corr_rate = []
            for sentense in range(len(X)):
                if np.random.random() < 0.1:
                    inp_data_sequence = [0] + X[sentense]
                    target_data_sequence = X[sentense] + [1]
                else:
                    inp_data_sequence = [0] + X[sentense][:-1]
                    target_data_sequence = X[sentense]

                self.session.run(self.train_op, feed_dict={self.tfX: inp_data_sequence, self.tfT: target_data_sequence})
                c, predict = self.session.run([self.cost, self.prediction], feed_dict={self.tfX: inp_data_sequence, self.tfT: target_data_sequence})

                costs[epoch] += c

                ''' calculate similarity of two sentences - real and predicted '''
                corr_rate.append(difflib.SequenceMatcher(None, predict, target_data_sequence).ratio())
            print("epoch: ", epoch, " cost: ", costs[epoch], " correct rate: ", sum(corr_rate)/len(corr_rate))

            if epoch == epochs-1:
                final_correct_rate = sum(corr_rate)/len(corr_rate)

            #if (epoch > 106) and (sum(corr_rate)/len(corr_rate) >= 0.91 or corr_rate[-1] == None or 
            # ( (sum(costs[epoch-5:epoch])/5 - sum(costs[epoch-105:epoch-100])/5) <  sum(costs[epoch-5:epoch])/100 )):
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
            name_dict = {LSTM : "LSTM", GateReccUnit : "GateReccUnit", RateReccUnit : "RateReccUnit", SimpleReccUnit : "SimpleReccUnit"}
            name = name_dict[recurr_unit] + optimizer + "_".join(map(str, optimizer_args)) + str(epochs) + "_reg_" + str(reg) + ".png"
            fig.savefig(name)

    def predict(self, prev_words):
        # don't use argmax, so that we can sample from this probability distribution
        self.session.run(tf.global_variables_initializer())
        return self.session.run(self.out_prob, feed_dict={self.tfX: prev_words})

    def save(self, filename):
        actual_params = self.session.run(self.params)
        np.savez(filename, *[p for Wb in actual_params for p in Wb])

    @staticmethod
    def load(filename, activation):
        ''' TODO: would prefer to save activation to file too '''
        npz = np.load(f(filename))
        param_num = len([elem for elem in npz])
        # [1, 2, 3, 4, 5, 6] => [(1, 2), (3, 4), (5, 6)]
        mylist = [npz['arr_' + str(n)] for n in range(param_num)]
        lst_W_b = list(zip(mylist[0::2], mylist[1::2]))

        V, D = npz['arr_0'].shape
        M2, _ = npz['arr_2'].shape

        unit = {4: SimpleReccUnit, 8: RateReccUnit, 12: GateReccUnit, 22: LSTM}
        #print("V: ", V, "D: ", D, "M2: ", M2)
        #print("lst_W_b[:2]: ", len(lst_W_b[:2]))
        #for i in lst_W_b[:2]:
        #    print(len(i))
        #print("lst_W_b[2:]: ", len(lst_W_b[2:]))
        #for i in lst_W_b[2:]:
        #    print(len(i))
        #print(unit[param_num-4])
        rnn = GeneratePoetry(D, M2, V)
        # self.W_embed (W and bias) + self.W_out (W and bias) = 4; num of recUnit params = length of npz - 4
        rnn.model_initializer(recurr_unit=unit[param_num-4], nonlin_func=activation, train_mode=False, lst_W_b=lst_W_b)

        return rnn

    def generate(self, word2index):
        index_to_word = {index:word for word, index in word2index.items()}
        words_id = len(index_to_word)

        words = [0]
        num_line = 0
        ''' it will be 4 lines '''
        while num_line < 4:
            choose_word_idx = np.random.choice(words_id, p=self.predict(words)[-1])
            words.append(choose_word_idx)
            ''' if  choose_word_idx != 0 or 1 - not start or end token'''
            if choose_word_idx > 1:
                print(index_to_word[choose_word_idx], end=" ")
                ''' end token '''
            elif choose_word_idx == 1:
                num_line += 1
                words = [0]
                print("")



if __name__ == "__main__":
    sentences, word2idx = get_poetry_data()
    print("num of sentences: ", len(sentences), "num of unique words: ", len(word2idx))
    #for rec_unit in [LSTM, GateReccUnit, RateReccUnit, SimpleReccUnit]:
    recurrent_model = GeneratePoetry(D=50, M2=50, V=len(word2idx))
    recurr_unit = LSTM
    session = tf.InteractiveSession()
    recurrent_model.set_session(session)
    recurrent_model.fit(X=sentences, recurr_unit=recurr_unit, nonlin_func="relu", optimizer="adam", optimizer_args=(1e-3, 0.9, 0.999), 
                        reg=0, epochs=500, show_fig=False)
    recurrent_model.save("LSTM.npz")

    loaded_recurrent_model = GeneratePoetry.load(filename="LSTM.npz", activation="relu")
    session = tf.InteractiveSession()
    loaded_recurrent_model.set_session(session)
    for i in range(5):
        loaded_recurrent_model.generate(word2index=word2idx)